CycleGAN：
input _nc：3 （输入通道数，RGB为3，灰度图像为1）
out_nc：输出通道数
ngf：G中最后一个卷积层的个数"of gen in the last conv layer"
ndf：D中第一个卷积层个数 "of D in the first layer"
netD：指定判别器结构，basic(70×70 patchGAN)、n layer(可以指定D的层数)、 pixel默认为basic
netG：指定生成器结构，resnet 9blocks、resnet 6blocks、unet256、unet 128，默认是resnet 9blocks
n_layer_D：netD为n layer时才使用
norm：默认为instance，instance normalization 或 batch normalization或 none
