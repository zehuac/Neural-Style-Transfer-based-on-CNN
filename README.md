### Neural Style Transfer 
This is a Tensorflow implementation for several techniques in the following papar:

["A Neural Algorithm of Artistic Style"](https://arxiv.org/pdf/1508.06576.pdf) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

["Image Style Transfer Using Convolutional Neural Networks"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

["Preserving Color in Neural Artistic Style Transfer"](https://arxiv.org/pdf/1606.05897.pdf) by Leon A. Gatys, Matthias Bethge, Aaron Hertzmann, Eli Shechtman

["Laplacian-Steered Neural Style Transfer"](https://arxiv.org/pdf/1707.01253.pdf) by Shaohua Li, Xinxing Xu, Liqiang Nie, Tat-Seng Chua

### Download VGG19 via following URL:
```
http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
```

### Commands

```
python run_main.py --content images/01.jpg --style images/starry-night.jpg --output stockholm.jpg
```
```
python run_main.py --content images/01.jpg --style images/starry-night.jpg --output stockholm.jpg --multi_style True --style_ratio 0.5 
```

### Parameters

| Para | Function  |
| ---------- | :-----------:  |
| --model_path  | 预训练的VGG模型的参数文件的路径    |
| --content  | 内容图片的路径    |
| --style  | 风格图片的路径    |
| --output  | 输出图片的路径    |
| --loss_ratio  | 内容损失和风格损失的相对权重，就是论文中公式7的alpha和beta    |
| --content_layers  | 用于计算内容损失的representation    |
| --style_layers  | 用于计算风格损失的representation，这个represention取自很多层，并对它们做一个加权和    |
| --content_layer_weights  | 内容represention的权重，实际上论文中和代码中都只采用了一个representation(4_2)，因此该权重只有一个    |
| --style_layer_weights  | 风格represention的权重，默认有五个，分别对应于每层挑选出来的用于计算风格损失的representation    |
| --initial_type  | 有三个选项，将初时待合成图片设置为内容图片或风格图片或随机白噪声图片(论文中的做法)    |
| --max_size  | 输入图片的最大尺寸，默认512    |
| --content_loss_norm_type  | 该选项用于确定内容损失函数normalization的方式，目测没啥用    |
| --num_iter  | 迭代次数，它默认设置的是1000，但我每次都跑了1000多次。。    |



以下是我加的：

| Para | Function  |
| ---------- | :-----------:  |
| --style2  | 第二张风格图片的路径    |
| --multi_style  | True/False 是否进行多风格迁移|
| --style_ratio  | 两个风格图片损失所占的比重    |
| --color_preseving  | True/False, 是否使用color preserving  |
| --color_convert_type  | yuv/ycrcb/luv/lab  |
| --color_preserve_algo  | 1/2, 1 直接替换亮度通道，2 使用论文中的那个公式 |
| --tv  | True/False, 是否添加tv项 |
| --laplace  | True/False, 是否添加laplace项 |
| --lap_lambda  | laplace项的系数 |
| --pooling_size  | ? |



