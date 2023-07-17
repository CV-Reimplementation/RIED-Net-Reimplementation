import torch
from .base_model import BaseModel
from . import networks
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from . import ssim
from .ssim import SSIM, MS_SSIM

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='ried_net', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_MS', type=float, default=10.0, help='weight for MS_SSIM loss')
            # parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight for VGG loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN_l', 'G_GAN_g', 'G_L1', 'D_real_l', 'D_real_g', 'D_fake_l', 'D_fake_g', 'MS_SSIM']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D_l', 'D_g']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD_l = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD_l,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)  # 70*70, basic
            self.netD_g = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD_g,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)  # 修改网络输出节点的参数

        if self.isTrain:
            # define loss functions
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.MS_SSIM = ssim.MS_SSIM().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_l = torch.optim.Adam(self.netD_l.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_g = torch.optim.Adam(self.netD_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_l)
            self.optimizers.append(self.optimizer_D_g)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake_l = self.netD_l(fake_AB.detach())  # local 和 global
        pred_fake_g = self.netD_l(fake_AB.detach())
        self.loss_D_fake_l = self.criterionGAN(pred_fake_l, False)
        self.loss_D_fake_g = self.criterionGAN(pred_fake_g, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real_l = self.netD_l(real_AB)  # 70*70
        pred_real_g = self.netD_g(real_AB)  # 128*128
        self.loss_D_real_l = self.criterionGAN(pred_real_l, True)
        self.loss_D_real_g = self.criterionGAN(pred_real_g, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake_l + self.loss_D_real_l + self.loss_D_fake_g + self.loss_D_real_g) * 0.5  # 修改
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_l = self.netD_l(fake_AB)
        pred_fake_g = self.netD_g(fake_AB)
        self.loss_G_GAN_l = self.criterionGAN(pred_fake_l, True)
        self.loss_G_GAN_g = self.criterionGAN(pred_fake_g, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Third perceptual loss
        # self.VGG_real = self.vgg(self.real_B.expand(
        #     [int(self.real_B.size()[0]), 3, int(self.real_B.size()[2]), int(self.real_B.size()[3])]))[0]
        # self.VGG_fake = self.vgg(self.fake_B.expand(
        #     [int(self.real_B.size()[0]), 3, int(self.real_B.size()[2]), int(self.real_B.size()[3])]))[0]
        # self.VGG_loss = self.criterionL1(self.VGG_fake, self.VGG_real) * self.opt.lambda_vgg
        # SSIM loss
        self.loss_MS_SSIM = (1-self.MS_SSIM(self.real_B, self.fake_B)) * self.opt.lambda_MS

        # combine loss and calculate gradients 考虑是否添加感知损失
        self.loss_G = self.loss_G_GAN_l + self.loss_G_GAN_g + self.loss_G_L1 + self.loss_MS_SSIM
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD_l, True)  # enable backprop for D
        self.optimizer_D_l.zero_grad()     # set D's gradients to zero
        self.set_requires_grad(self.netD_g, True)  # enable backprop for D
        self.optimizer_D_g.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        self.optimizer_D_l.step()          # update D's weights
        self.optimizer_D_g.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD_l, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_g, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


    # def get_current_errors(self):
    #     return OrderedDict([('G_GAN_l', self.loss_G_GAN_l.data[0]),
    #                         ('G_GAN_g', self.loss_G_GAN_g.data[0]),
    #                         ('G_L1', self.loss_G_L1.data[0]),
    #                         # ('G_VGG', self.VGG_loss.data[0]),
    #                         ('D_real_l', self.loss_D_real_l.data[0]),
    #                         ('D_fake_l', self.loss_D_fake_l.data[0]),
    #                         ('D_real_g', self.loss_D_real_g.data[0]),
    #                         ('D_fake_g', self.loss_D_fake_g.data[0]),
    #                         ('MS_SSIM', self.loss_MS_SSIM[0])
    #                         ])
    #
    # def get_current_visuals(self):
    #     real_A = util.tensor2im(self.real_A.data)
    #     fake_B = util.tensor2im(self.fake_B.data)
    #     real_B = util.tensor2im(self.real_B.data)
    #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
    #
    # def save(self, label):
    #     self.save_network(self.netG, 'G', label, self.gpu_ids)
    #     self.save_network(self.netD_l, 'D_l', label, self.gpu_ids)
    #     self.save_network(self.netD_g, 'D_g', label, self.gpu_ids)


#Extracting VGG feature maps before the 2nd maxpooling layer
# class VGG16(torch.nn.Module):
#     def __init__(self):
#         super(VGG16, self).__init__()
#         vgg_pretrained_features = models.vgg16(pretrained=True).features
#         self.stage1 = torch.nn.Sequential()
#         self.stage2 = torch.nn.Sequential()
#         for x in range(4):
#             self.stage1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.stage2.add_module(str(x), vgg_pretrained_features[x])
#         for param in self.parameters():
#             param.requires_grad = False
#     def forward(self, X):
#         h_relu1 = self.stage1(X)
#         h_relu2 = self.stage2(h_relu1)
#         return h_relu2