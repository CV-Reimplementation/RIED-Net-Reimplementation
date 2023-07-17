import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


# from util.image_pool import ImagePool
# from .base_model import BaseModel
# from . import networks


class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x)


class Residual_Inception(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels, scale=1.0):
		super(Residual_Inception, self).__init__()
		self.scale = scale
		self.branch0 = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1)
		self.branch1 = nn.Sequential(BasicConv2d(in_channels, mid_channels, kernel_size=3, stride=1),
		                             BasicConv2d(mid_channels, out_channels, kernel_size=3, stride=1))

	# self.conv2d = nn.Conv2d(kernel_size=1, stride=1)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.relu(out)
		return out


class UnetDown(nn.Module):
	def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
		super(UNetDown, self).__init__()
		layers = [nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, bias=False)]
		if normalize:
			layers.append(nn.InstanceNorm2d(out_size))
		layers.append(nn.ReLU(inplace=True))
		if dropout:
			layers.append(nn.Dropout(dropout))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)


class UnetUp(nn.Module):
	def __init__(self, in_size, out_size, dropout=0.0):
		super(UNetUp, self).__init__()
		layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, bias=False),
		          nn.InstanceNorm2d(out_size),
		          nn.ReLU(inplace=True)]
		if dropout:
			layers.append(nn.Dropout(dropout))
		self.model = nn.Sequential(*layers)

	def forward(self, x, skip_input):
		x = self.model(x)
		x = torch.cat((x, skip_input), 1)
		return x


class Unet(nn.Module):
	def __init__(self, in_channels=3, out_channels=3):
		super(Unet, self).__init__()
		self.RI1 = Residual_Inception(1, 32, 32)
		self.RI2 = Residual_Inception(32, 64, 64)
		self.RI3 = Residual_Inception(64, 128, 128)
		self.RI4 = Residual_Inception(128, 256, 256)
		self.RI5 = Residual_Inception(256, 512, 512)
		self.RI6 = Residual_Inception(512, 256, 256)
		self.RI7 = Residual_Inception(256, 128, 128)
		self.RI8 = Residual_Inception(128, 64, 64)
		self.RI9 = Residual_Inception(64, 32, 32)

		self.down1 = UnetDown(32, 32)
		self.down2 = UnetDown(64, 64)
		self.down3 = UnetDown(128, 128)
		self.down4 = UnetDown(256, 256)

		self.up1 = UnetUp(512, 512)
		self.up2 = UnetUp(256, 256)
		self.up3 = UnetUp(128, 128)
		self.up4 = UnetUp(64, 64)

		self.final = nn.Conv2d(32, 1, kernel_size=1, stride=1)

	def forward(self, x):
		r1 = self.RI1(x)
		d1 = self.down1(r1)
		r2 = self.RI2(d1)
		d2 = self.down(r2)
		r3 = self.RI3(d2)
		d3 = self.down(r3)
		r4 = self.RI4(d3)
		d4 = self.down(r4)
		r5 = self.RI5(d4)
		u1 = self.up(r5, r1)
		r6 = self.RI6(u1)
		u2 = self.up(r6, r2)
		r7 = self.RI7(u2)
		u3 = self.up(r7, r3)
		r8 = self.RI8(u3)
		u4 = self.up(r8, r4)
		r9 = self.RI9(u4)

		return self.final(r9)



if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	image_size = 128
	batch_size = 1
	dataroot = "C:/Users/GAN/Documents/DCGAN/dataroot/A/test"
	num_workers = 2
	dataset = torchvision.datasets.ImageFolder(root=dataroot, transform=transforms.Compose([
		transforms.Resize(image_size),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	]))
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	input =
	net = Unet().to(device)
	Unet(3, 3, 8, 64, norm_layer=False, use_dropout=True)

#  无情的分界线
#
#
# class FeedGANModel(BaseModel):
#
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         """Add new dataset-specific options, and rewrite default values for existing options.
#
#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
#
#         Returns:
#             the modified parser.
#
#         For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
#         A (source domain), B (target domain).
#         Generators: G_A: A -> B; G_B: B -> A.
#         Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
#         Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
#         Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
#         Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
#         Dropout is not used in the original CycleGAN paper.
#         """
#         parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
#         if is_train:
#             parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
#             parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
#             parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
#
#         return parser
#
#     def __init__(self, opt):
#         """Initialize the FeedGAN class.
#
#         Parameters:
#             opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseModel.__init__(self, opt)
#         # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
#         self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
#         # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
#         self.visual_names = ['real_A', 'fake_B', 'real_B']
#         # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
#         if self.isTrain:
#             self.model_names = ['G', 'D']
#         else:  # during test time, only load G
#             self.model_names = ['G']
#         # define networks (both generator and discriminator)
#         self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
#                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#
#         if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
#             self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
#                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
#
#         if self.isTrain:
#             # define loss functions
#             self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
#             self.criterionL1 = torch.nn.L1Loss()
#             # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
#             self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_G)
#             self.optimizers.append(self.optimizer_D)
#
#     def set_input(self, input):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.
#
#         Parameters:
#             input (dict): include the data itself and its metadata information.
#
#         The option 'direction' can be used to swap images in domain A and domain B.
#         """
#         AtoB = self.opt.direction == 'AtoB'  # True
#         self.real_A = input['A' if AtoB else 'B'].to(self.device)
#         self.real_B = input['B' if AtoB else 'A'].to(self.device)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']
#
#     def forward(self):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         self.fake_B = self.netG(self.real_A)  # G(A)
#
#     def backward_D(self):
#         """Calculate GAN loss for the discriminator"""
#         # Fake; stop backprop to the generator by detaching fake_B
#         fake_AB = torch.cat((self.real_A, self.fake_B),
#                             1)  # we use conditional GANs; we need to feed both input and output to the discriminator
#         pred_fake = self.netD(fake_AB.detach())
#         self.loss_D_fake = self.criterionGAN(pred_fake, False)
#         # Real
#         real_AB = torch.cat((self.real_A, self.real_B), 1)
#         pred_real = self.netD(real_AB)
#         self.loss_D_real = self.criterionGAN(pred_real, True)
#         # combine loss and calculate gradients
#         self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
#         self.loss_D.backward()
#
#     def backward_G(self):
#         """Calculate GAN and L1 loss for the generator"""
#         # First, G(A) should fake the discriminator
#         fake_AB = torch.cat((self.real_A, self.fake_B), 1)
#         pred_fake = self.netD(fake_AB)
#         self.loss_G_GAN = self.criterionGAN(pred_fake, True)
#         # Second, G(A) = B
#         self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
#         # combine loss and calculate gradients
#         self.loss_G = self.loss_G_GAN + self.loss_G_L1
#         self.loss_G.backward()
#
#     def optimize_parameters(self):
#         self.forward()  # compute fake images: G(A)
#         # update D
#         self.set_requires_grad(self.netD, True)  # enable backprop for D
#         self.optimizer_D.zero_grad()  # set D's gradients to zero
#         self.backward_D()  # calculate gradients for D
#         self.optimizer_D.step()  # update D's weights
#         # update G
#         self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
#         self.optimizer_G.zero_grad()  # set G's gradients to zero
#         self.backward_G()  # calculate graidents for G
#         self.optimizer_G.step()  # udpate G's weights
#
#
# import torch
# import torchvision
# import torch.nn as nn
# import torchvision.utils as vutils
# from torchvision.transforms import transforms
# from Resnet import ResNet
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# image_size = 64
# batch_size = 128
# dataroot = "img_align_celeba"
# num_workers = 2
# dataset = torchvision.datasets.ImageFolder(root=dataroot, transform=transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
#
# def weight_init(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         # str.find("字符串")如果找到，则返回该字符串的索引值，如果找不到，则返回-1。
#         # 这里指 如果包含"Conv"则执行if
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BathNorm") != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
#
#
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             #  class torch.nn.Conv2d(in_channels, out_channels,
#             #  kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#             nn.Linear(4, 4, 16),
#             ResNet()
#
#
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),  # 对小批量的3d或4d输入进行批标准化操作
#             nn.ReLU(True),
#             # state size:ngf*8, 4, 4
#             nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),  # 对小批量的3d或4d输入进行批标准化操作
#             nn.ReLU(True),
#             # state size:ngf*4, 8, 8
#             nn.ConvTranspose2d(ngf*4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),  # 对小批量的3d或4d输入进行批标准化操作
#             nn.ReLU(True),
#             # state size:ngf*2, 16, 16
#             nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),  # 对小批量的3d或4d输入进行批标准化操作
#             nn.ReLU(True),
#             # state size:ngf, 32, 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size:nc, 64, 64
#         )
#
#     def forward(self, x):
#         return self.main(x
