import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class RIEDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='ried_net', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1']  # MAE，平均绝对误差，即L1损失
        self.visual_names = ['fake_B']

        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks RIED-net只有一个生成器，不进行生成对抗训练
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 学习率默认为0.002，学习衰减率为0.005
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'  # True
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # G(A) = B
        self.loss_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1  # 100
        # combine loss and calculate gradients
        self.loss_G = self.loss_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights





# class BasicConv2d(nn.Module):
# 	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
# 		super(BasicConv2d, self).__init__()
# 		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
# 		self.bn = nn.BatchNorm2d(out_channels)
#
# 	def forward(self, x):
# 		x = self.conv(x)
# 		x = self.bn(x)
# 		return F.relu(x)
#
#
# class Residual_Inception(nn.Module):
# 	def __init__(self, in_channels, out_channels, scale=1.0):
# 		super(Residual_Inception, self).__init__()
# 		self.scale = scale
# 		self.branch0 = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1)
# 		self.branch1 = nn.Sequential(BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1),
# 		                             BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1))
#
# 	# self.conv2d = nn.Conv2d(kernel_size=1, stride=1)
#
# 	def forward(self, x):
# 		x0 = self.branch0(x)
# 		x1 = self.branch1(x)
# 		out = torch.cat((x0, x1), 1)
# 		out = self.relu(out)
# 		return out
#
# class UnetGenerator(nn.Module):
# 	"""Create a Unet-based generator"""
#
# 	def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
# 		"""Construct a Unet generator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             output_nc (int) -- the number of channels in output images
#             num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
#                                 image of size 128x128 will become of size 1x1 # at the bottleneck
#             ngf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#
#         We construct the U-Net from the innermost layer to the outermost layer.
#         It is a recursive process.
#         """
# 		super(UnetGenerator, self).__init__()  # 默认ngf=64
# 		# construct unet structure
# 		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, ngf * 8, input_nc=None, submodule=None,
# 		                                     norm_layer=norm_layer, innermost=True)  # add the innermost layer
# 		print("第一个unet块是：", unet_block)
# 		for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
# 			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
# 			                                     norm_layer=norm_layer, use_dropout=use_dropout)
# 			print("for循环中第%d个unet块是：%s", i, unet_block)
# 		# gradually reduce the number of filters from ngf * 8 to ngf
# 		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
# 		                                     norm_layer=norm_layer)
# 		print("for循环外第1个unet块是：", unet_block)
# 		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
# 		                                     norm_layer=norm_layer)
# 		print("for循环外第2个unet块是：", unet_block)
# 		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
# 		print("for循环外第个unet块是：", unet_block)
# 		self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
# 		                                     norm_layer=norm_layer)  # add the outermost layer
# 		print("整体模型是：", self.model)
#
# 	def forward(self, input):
# 		return self.model(input)
#
#
# class UnetSkipConnectionBlock(nn.Module):
# 	def __init__(self, outer_nc, inner_nc, input_nc=None,
# 	             submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
# 		super(UnetSkipConnectionBlock, self).__init__()
# 		self.outermost = outermost
# 		#  use_bias = norm_layer.func == nn.InstanceNorm2d
# 		use_bias = True
# 		if input_nc is None:
# 			input_nc = outer_nc
# 		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
# 		                     stride=2, padding=1, bias=use_bias)
# 		downrelu = nn.ReLU(0.2)
# 		# downnorm = norm_layer(inner_nc)
#
# 		uprelu = nn.ReLU(True)
# 		# upnorm = norm_layer(outer_nc)
#
# 		if outermost:
# 			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
# 			                            kernel_size=3, stride=2,
# 			                            padding=1)
# 			down = [downconv]
# 			up = [uprelu, upconv, nn.Tanh()]
# 			model = down + [submodule] + up
# 		elif innermost:
# 			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
# 			                            kernel_size=3, stride=2,
# 			                            padding=1, bias=use_bias)
# 			down = [downrelu, downconv]
# 			RI1 = Residual_Inception(input_nc, inner_nc)
# 			RI = [RI1]
# 			up = [uprelu, upconv]
# 			model = down + RI + up
# 		else:
# 			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
# 			                            kernel_size=3, stride=2,
# 			                            padding=1, bias=use_bias)
# 			RI1 = Residual_Inception(input_nc, inner_nc)
# 			RI2 = Residual_Inception(input_nc, inner_nc)
# 			RI1 = [RI1]
# 			RI2 = [RI2]
# 			down = [downrelu, downconv]
# 			up = [uprelu, upconv]
# 			if use_dropout:
# 				model = down + RI1 + [submodule] + RI2 + up + [nn.Dropout(0.5)]
# 			else:
# 				model = down + RI1 + [submodule] + RI2 + up
#
# 		self.model = nn.Sequential(*model)
#
# 	def forward(self, x):
# 		if self.outermost:
# 			return self.model(x)
# 		else:  # add skip connections
# 			return torch.cat([x, self.model(x)], 1)  # 连接x与model(x)，dim=1表示按列连接



# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return nn.functional.relu(x)
#
# class Branch0(nn.Module):
#     def __init__(self, in_channels, out_channels, scale=1.0):
#         super(Branch0, self).__init__()
#         self.scale = scale
#         self.branch0 = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         return x0
#
# class Branch1(nn.Module):
#     def __init__(self, in_channels, out_channels, scale=1.0):
#         super(Branch1, self).__init__()
#         self.branch1 = nn.Sequential(BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1),
#                                      BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1))
#
#     def forward(self, x):
#         x1 = self.branch1(x)
#         # out = torch.cat((x0, x1), 1)
#         # out = self.relu(out)
#         return x1
#
# class Residual_Inception(nn.Module):
#     def __init__(self, in_channels, out_channels, scale=1.0):
#         super(Residual_Inception, self).__init__()
#         self.RI0 = Branch0(in_channels, out_channels)
#         self.RI1 = Branch1(in_channels, out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x0 = self.RI0(x)
#         x1 = self.RI1(x)
#         out = torch.cat((x0, x1), 1)
#         out = self.relu(out)
#         return out
#
#
# class UnetGenerator(nn.Module):
#     """Create a Unet-based generator"""
#
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet generator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             output_nc (int) -- the number of channels in output images
#             num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
#                                 image of size 128x128 will become of size 1x1 # at the bottleneck
#             ngf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#
#         We construct the U-Net from the innermost layer to the outermost layer.
#         It is a recursive process.
#         """
#         super(UnetGenerator, self).__init__()
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer,
#                                              innermost=True)  # add the innermost layer
#         # for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
#         #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
#         #                                          norm_layer=norm_layer, use_dropout=use_dropout)
#         # gradually reduce the number of filters from ngf * 8 to ngf
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
#                                              norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(int(ngf/2), ngf, input_nc=None, submodule=unet_block,
#                                              norm_layer=norm_layer)
#         # unet_block = UnetSkipConnectionBlock(int(ngf/2), ngf, input_nc=None, submodule=unet_block,
#         #                                      norm_layer=norm_layer)
#         self.model = UnetSkipConnectionBlock(3, int(ngf/2), input_nc=input_nc, submodule=unet_block, outermost=True,
#                                              norm_layer=norm_layer)  # add the outermost layer
#         print("模型结构是", self.model)
#
#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)
#
#
# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """
#
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet submodule with skip connections.
#
#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#         """
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#
#         # if type(norm_layer) == functools.partial:
#         #     use_bias = norm_layer.func == nn.InstanceNorm2d
#         # else:
#         #     use_bias = norm_layer == nn.InstanceNorm2d
#         use_bias = True
#         if input_nc is None:
#             input_nc = outer_nc
#         self.branch = Branch1(input_nc, inner_nc)
#         downconv = nn.Conv2d(input_nc, input_nc, kernel_size=3,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.ReLU()
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(inner_nc)
#         print("input_nc是", input_nc)
#         print("inner_nc是", inner_nc)
#         print("outer_nc是", outer_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                         kernel_size=3, stride=2,
#                                         padding=1, bias=use_bias)
#             RI1 = Residual_Inception(input_nc, inner_nc)
#             RI2 = Residual_Inception(inner_nc, inner_nc)
#             RI3 = Residual_Inception(inner_nc, inner_nc)
#             down = [downconv]
#             up = [uprelu, upconv, nn.Tanh()]
#             final = nn.Conv2d(inner_nc, 3, kernel_size=1)
#             final = [RI3] + [final]
#             model = [RI1] + down + [submodule] + up + [RI2] + final
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                         kernel_size=3, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             RI1 = Residual_Inception(input_nc, inner_nc)
#             RI2 = Residual_Inception(input_nc, inner_nc)
#             up = [uprelu, upconv]
#             model = [RI1] + down + [RI2] + up
#
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=3, stride=2,
#                                         padding=1, bias=use_bias)
#             RI1 = Residual_Inception(input_nc, inner_nc)
#             RI2 = Residual_Inception(input_nc, inner_nc)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv]
#             if use_dropout:
#                 model = [RI1] + down + [submodule] + [RI2] + up + [nn.Dropout(0.5)]
#             else:
#                 model = [RI1] + down + [submodule] + [RI2] + up
#
#         self.model = nn.Sequential(*model)
#         print(self.model)
#
#     def forward(self, x):
#         # b1 = self.branch(x)
#         if self.innermost:
#             return self.model(x)
#         else:  # add skip connections
#             return torch.cat([b1, self.model(x)], 1)  # 连接x与model(x)，dim=1表示按列连接
#
#
# if __name__ == '__main__':
#     net = UnetGenerator(input_nc=1, output_nc=32, num_downs=7)