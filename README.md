### Neural Style Transfer 
This is a Tensorflow implementation for several techniques in the following papers:

["A Neural Algorithm of Artistic Style"](https://arxiv.org/pdf/1508.06576.pdf) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

["Image Style Transfer Using Convolutional Neural Networks"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

["Preserving Color in Neural Artistic Style Transfer"](https://arxiv.org/pdf/1606.05897.pdf) by Leon A. Gatys, Matthias Bethge, Aaron Hertzmann, Eli Shechtman

["Laplacian-Steered Neural Style Transfer"](https://arxiv.org/pdf/1707.01253.pdf) by Shaohua Li, Xinxing Xu, Liqiang Nie, Tat-Seng Chua

### Setup
This code is based on Tensorflow. It has been tested on Windows 10 and Ubuntu 16.04 LTS.  
Dependencies:  
* Tensowflow
* nmupy, matplotlib, scipy, PIL  

Recommended but optional:  
* CUDA
* cudnn

### Download VGG19 via following URL:
```
http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
```
And please save the file `imagenet-vgg-verydeep-19.mat` under ``pre_trained_model``

### Commands
  Quick start
```
python run_main.py --content images/01.jpg --style images/starry-night.jpg --output result.jpg
```
  Multi-Style transfer
```
python run_main.py --content images/01.jpg --style images/starry-night.jpg --output result_multi-style.jpg --multi_style True --style_ratio 0.5 
```
  Color Preserving 
```
python run_main.py --content images/01.jpg --style images/starry-night.jpg --output result_color-preserve_alg-2.jpg --color_preseving True --color_preserve_algo 2
```
  Add Laplacian Loss
```
python run_main.py --content images/01.jpg --style images/starry-night.jpg --output result_lap_4.jpg --laplace True --pooling_size 4 
```
### Arguments

Required:  

* ``--content``: Filename of the content image. *Default:* ``images/tubingen.jpg``
* ``--style``: Filename of the style image. *Default:* ``images/starry-night.jpg``
* ``--output``: Filename of the output image. *Default:* ``result.jpg``

Optional:
* ``--model_path``: Path to the pre-trained VGG model. *Default:* ``pre_trained_model``  
* ``--loss_ratio``: Weight of content-loss relative to style-loss, the alpha over beta in the paper. *Default:* ``0.5``  
* ``--content_layers``: Space-separated VGG-19 layer names used for content loss computation. *Default:* ``conv4_2``  
* ``--style_layers``: Space-separated VGG-19 layer names used for style loss computation. *Default:* ``relu1_1, relu2_1, relu3_1, relu4_1, relu5_1``  
* ``--content_layer_weights``: Space-separated weights of each content layer to the content loss. *Default:* ``1.0``
* ``--style_layer_weights``: Space-separated weights of each style layer to loss. *Default:* ``0.2 0.2 0.2 0.2 0.2``
* ``--initial_type``: The initial image for optimization. (notation in the paper : x) Choices: content, style, random. *Default:* ``content``
* ``--max_size``: Maximum width or height of the input images. *Default:* ``512``
* ``--content_loss_norm_type``: Different types of normalization for content loss. *Choices:* [1](https://arxiv.org/pdf/1508.06576v2.pdf), [2](https://arxiv.org/pdf/1604.08610.pdf), [3](https://github.com/cysmith/neural-style-tf). *Default:* ``3``
* ``--num_iter``: The number of iterations to run. *Default:* ``1000``
* ``--multi_style``: If to use multiple style transfer. *Choices:* ``True/False`` *Default:* ``False``
* ``--style2``: Filename to use multiple style images. *Default:* ``images/kandinsky.jpg``
* ``--style_ratio``: The ratio between two styles. *Default:* ``0.5``
* ``--color_preseving``: If to use color preserving. *Choices:* ``True/False`` *Default:* ``False``
* ``--color_convert_type``: Color spaces (YUV, YCrCb, CIE L\*u\*v\*, CIE L\*a\*b\*)  for luminance-matching conversion to original colors. *Choices:* ``yuv``, ``ycrcb``, ``luv``, `lab`. *Default*: ``yuv``
* ``--color_preserve_algo``: Color preserving algorithm. *Choices:* [1](https://github.com/cysmith/neural-style-tf), [2](https://arxiv.org/pdf/1606.05897.pdf) The Approach #2,  *Default:* `1`
* `--tv`: If to add the total variational loss function. *Choices:* ``True/False`` *Default:* ``False``
* `--laplace` If to add the [Laplacian loss](https://arxiv.org/pdf/1707.01253.pdf). *Choices:* ``True/False`` *Default:* ``False``
* `--lap_lambda` Weight for the Laplacian loss. *Default:* ``100``
* `--pooling_size` The filter size for pooling layer before using Laplacian operator. *choices:* `4`, `16`, `20` (choosing 20 means to use the [multiple Laplacians](https://arxiv.org/pdf/1707.01253.pdf) with 4x4 and 16x16 pooling layer) Default:* `4`

If you find this code useful for your research, please cite:

```
@misc{NST2018,
  author = {Zehua Chen, Jiaying Yang, Yiping Xie},
  title = {DL_Project},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Chen-Z-H/DL_Project},
}
```

### Acknowledgements

The implementation is based on the project:
* Tensorflow implementation 'neural-style-tf' by [cysmith](https://github.com/cysmith/neural-style-tf)
* Tensorflow implementation 'tensorflow-style-transfer' by [hwalsuklee](https://github.com/hwalsuklee/tensorflow-style-transfer)



