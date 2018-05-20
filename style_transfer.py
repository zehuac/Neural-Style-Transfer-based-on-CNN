# coding=UTF-8
import tensorflow as tf
import numpy as np
import collections
import cv2
import vgg19


def preprocess(img):
    # bgr to rgb
    img = img[..., ::-1]
    # shape (h, w, d) to (1, h, w, d)
    img = img[np.newaxis, :, :, :]
    img -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return img


def postprocess(img):
    img += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    # shape (1, h, w, d) to (h, w, d)
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    # rgb to bgr
    img = img[..., ::-1]
    return img

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0., -1., 0.],
                           [-1., 4., -1.],
                           [0., -1.0, 0.]])
  return simple_conv(x, laplace_k)


class StyleTransfer:

    def __init__(self, content_layer_ids, style_layer_ids, init_image, content_image,
                 style_image, session, net, num_iter, loss_ratio, content_loss_norm_type,
                 style_image2=np.array([]), style_ratio=0.5, multi_style=False, color_convert_type="yuv",
                 color_preserve=False, laplace=False, lap_lambda=10, tv=False):

        self.net = net
        self.sess = session

        self.multi_style = multi_style
        # sort layers info
        self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))
        self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layer_ids.items()))

        # preprocess input images
        self.content_img = content_image
        self.p0 = np.float32(self.net.preprocess(content_image))
        self.a0 = np.float32(self.net.preprocess(style_image))
        self.x0 = np.float32(self.net.preprocess(init_image))

        if self.multi_style == True:
            self.b0 = np.float32(self.net.preprocess(style_image2))
            self.sr = style_ratio
        else:
            self.b0 = []

        # parameters for optimization
        self.content_loss_norm_type = content_loss_norm_type
        self.num_iter = num_iter
        self.loss_ratio = loss_ratio

        # switches
        self.color_convert_type = color_convert_type        # Used for color
        self.color_preserve = color_preserve
        self.laplace = laplace
        self.lap_lambda = lap_lambda
        self.tv = tv

        # build graph for style transfer
        self._build_graph()

    def _build_graph(self):

        """ prepare data """
        # this is what must be trained
        self.x = tf.Variable(self.x0, trainable=True, dtype=tf.float32)

        # graph input
        self.p = tf.placeholder(tf.float32, shape=self.p0.shape, name='content')
        self.a = tf.placeholder(tf.float32, shape=self.a0.shape, name='style')

        # get content-layer-feature for content loss
        content_layers = self.net.feed_forward(self.p, scope='content')     # 这里得到内容图像网络每一层的输出
        self.Ps = {}
        for id in self.CONTENT_LAYERS:
            self.Ps[id] = content_layers[id]

        # get style-layer-feature for style loss
        style_layers = self.net.feed_forward(self.a, scope='style')
        self.As = {}
        for id in self.STYLE_LAYERS:
            self.As[id] = self._gram_matrix(style_layers[id])

        # get style loss of the second style image
        if self.multi_style == True:
            self.b = tf.placeholder(tf.float32, shape=self.b0.shape, name='style2')
            style_layers = self.net.feed_forward(self.b, scope='style2')
            self.Bs = {}
            for id in self.STYLE_LAYERS:
                self.Bs[id] = self._gram_matrix(style_layers[id])

        
        # get layer-values for x
        self.Fs = self.net.feed_forward(self.x, scope='mixed')      # 这里得到合成图像网络每一层的输出

        """ compute loss """
        L_content = 0
        L_style = 0
        L_laplacian=0

        for id in self.Fs:
            if id in self.CONTENT_LAYERS:
                ## content loss ##

                F = self.Fs[id]            # content feature of x
                P = self.Ps[id]            # content feature of p

                _, h, w, d = F.get_shape() # first return value is batch size (must be one)
                N = h.value*w.value        # product of width and height
                M = d.value                # number of filters

                w = self.CONTENT_LAYERS[id] # weight for this layer

                # You may choose different normalization constant
                if self.content_loss_norm_type == 1:
                    L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / 2 # original paper
                elif self.content_loss_norm_type == 2:
                    L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / (N*M) #artistic style transfer for videos
                elif self.content_loss_norm_type == 3: # this is from https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
                    L_content += w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((F - P), 2))

            elif id in self.STYLE_LAYERS:
                ## style loss ##

                F = self.Fs[id]

                _, h, w, d = F.get_shape()  # first return value is batch size (must be one)
                N = h.value * w.value       # product of width and height
                M = d.value                 # number of filters

                w = self.STYLE_LAYERS[id]   # weight for this layer

                G = self._gram_matrix(F)    # style feature of x
                A = self.As[id]             # style feature of a

                L_style += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G-A), 2))


                if self.multi_style == True:
                    B = self.Bs[id]  # style feature of second style (if exists)
                    L_style2 = w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G-B), 2))
                    L_style = self.sr * L_style + (1 - self.sr) * L_style2


        # fix beta as 1
        alpha = self.loss_ratio
        beta = 1

        L_laplacian = 0
        if self.laplace:
            # laplacian loss

            L_1 = laplace(vgg19._avgpool_layer(self.p0)[0, :, :, 0]) + \
                  laplace(vgg19._avgpool_layer(self.p0)[0, :, :, 1]) + \
                  laplace(vgg19._avgpool_layer(self.p0)[0, :, :, 2])
            L_2 = laplace(vgg19._avgpool_layer(self.x)[0, :, :, 0]) + \
                  laplace(vgg19._avgpool_layer(self.x)[0, :, :, 1]) + \
                  laplace(vgg19._avgpool_layer(self.x)[0, :, :, 2])
            # L_1 = laplace(self.p0[0][:][:][0])
            # L_2 = laplace(self.x)
            L_laplacian = self.lap_lambda * tf.reduce_sum(tf.pow(L_1 - L_2, 2))

        l_tv = 0
        if self.tv:
            _, h, w, d = self.content_img.shape
            c = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))
            l_tv = tf.image.total_variation(c)

        self.L_content = L_content
        self.L_style = L_style
        self.L_total = alpha*L_content + beta*L_style + 1e-3 * l_tv + L_laplacian

    def update(self):
        """ define optimizer L-BFGS """
        # this call back function is called every after loss is updated
        global _iter
        _iter = 0
        def callback(tl, cl, sl):
            global _iter
            print('iter : %4d, ' % _iter, 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
            _iter += 1

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.L_total, method='L-BFGS-B', options={'maxiter': self.num_iter})

        """ session run """
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # optmization
        if self.multi_style == True:
            print("Multi-style transfer")
            optimizer.minimize(self.sess, feed_dict={self.a: self.a0, self.b: self.b0, self.p: self.p0},
                               fetches=[self.L_total, self.L_content, self.L_style], loss_callback=callback)
        else:
            print("Single style transfer")
            optimizer.minimize(self.sess, feed_dict={self.a: self.a0, self.p: self.p0},
                           fetches=[self.L_total, self.L_content, self.L_style], loss_callback=callback)

        """ get final result """
        final_image = self.sess.run(self.x)


        # ensure the image has valid pixel-values between 0 and 255
        final_image = np.clip(self.net.undo_preprocess(final_image), 0.0, 255.0)


        if self.color_preserve:
            # color presevering
            final_image = self.convert_to_original_colors(np.copy(self.content_img), final_image)

        return final_image

    def _gram_matrix(self, tensor):

        shape = tensor.get_shape()

        # Get the number of feature channels for the input tensor,
        # which is assumed to be from a convolutional layer with 4-dim.
        num_channels = int(shape[3])

        # Reshape the tensor so it is a 2-dim matrix. This essentially
        # flattens the contents of each feature-channel.
        matrix = tf.reshape(tensor, shape=[-1, num_channels])

        # Calculate the Gram-matrix as the matrix-product of
        # the 2-dim matrix with itself. This calculates the
        # dot-products of all combinations of the feature-channels.
        gram = tf.matmul(tf.transpose(matrix), matrix)

        return gram


    def convert_to_original_colors(self, content_img, stylized_img):
        content_img = postprocess(content_img)
        stylized_img = postprocess(stylized_img)
        if self.color_convert_type == 'yuv':
            cvt_type = cv2.COLOR_BGR2YUV
            inv_cvt_type = cv2.COLOR_YUV2BGR
        elif self.color_convert_type == 'ycrcb':
            cvt_type = cv2.COLOR_BGR2YCR_CB
            inv_cvt_type = cv2.COLOR_YCR_CB2BGR
        elif self.color_convert_type == 'luv':
            cvt_type = cv2.COLOR_BGR2LUV
            inv_cvt_type = cv2.COLOR_LUV2BGR
        elif self.color_convert_type == 'lab':
            cvt_type = cv2.COLOR_BGR2LAB
            inv_cvt_type = cv2.COLOR_LAB2BGR
        content_cvt = cv2.cvtColor(content_img, cvt_type)
        stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
        c1, _, _ = cv2.split(stylized_cvt)
        _, c2, c3 = cv2.split(content_cvt)
        merged = cv2.merge((c1, c2, c3))
        dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
        dst = preprocess(dst)
        return dst










