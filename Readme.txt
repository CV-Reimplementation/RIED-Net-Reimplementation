CycleGAN��
input _nc��3 ������ͨ������RGBΪ3���Ҷ�ͼ��Ϊ1��
out_nc�����ͨ����
ngf��G�����һ�������ĸ���"of gen in the last conv layer"
ndf��D�е�һ���������� "of D in the first layer"
netD��ָ���б����ṹ��basic(70��70 patchGAN)��n layer(����ָ��D�Ĳ���)�� pixelĬ��Ϊbasic
netG��ָ���������ṹ��resnet 9blocks��resnet 6blocks��unet256��unet 128��Ĭ����resnet 9blocks
n_layer_D��netDΪn layerʱ��ʹ��
norm��Ĭ��Ϊinstance��instance normalization �� batch normalization�� none
