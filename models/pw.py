import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms import transforms

"""
img = cv2.imread("/home/qian/XCC/pytorch-CycleGAN-and-pix2pix-master1/models/9.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
plt.figure("二维小波一级变换")
coeffs = pywt.dwt2(img, 'haar')  # 离散小波变换
cA, (cH, cV, cD) = coeffs
# rfm = DWTForward(J=3, mode='zero', wave='db3')
# xl, xh = rfm(img)
# 将各子图拼接
# cA = cA + 255
cH = cH + 255
cV = cH + 255
cD = cD + 255
AH = np.concatenate([cA, cH], axis=1)
VD = np.concatenate([cV, cD], axis=1)
img = np.concatenate([AH, VD], axis=0)
plt.imshow(AH, 'gray')
plt.show()
"""

img = Image.open('/home/qian/XCC/pytorch-CycleGAN-and-pix2pix-master1/models/9.jpg').convert('RGB')
trans = transforms.Compose([transforms.ToTensor()])

img = trans(img)
img = img.unsqueeze(0)

# img.show()

# if grayscale:
#     transform_list += [transforms.Normalize((0.5,), (0.5,))]
# else:
#     transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# img_tensor = torch.transforms.To_Tensor()
# a = transforms.ToPILImage(a)
# trans = transforms.Compose(transforms.Resize((128)))






# 显示为灰度图
# plt.imshow(img, 'gray')
# # plt.title('result')
# plt.show()



xfm = DWTForward(J=1, mode='reflect', wave='haar')  # Accepts all wave types available to PyWavelets  # mode='zero', 'symmetric', reflect', 'periodization'
ifm = DWTInverse(mode='zero', wave='haar')  # wave = 'db3', 'haar',
h = xfm(img)
xl = h[0]
xh = h[1]
xh1 = xh[0][:, :, 0, :, :]
xh2 = xh[0][:, :, 1, :, :]
xh3 = xh[0][:, :, 2, :, :]

b = ToPILImage()(xl.squeeze(0))
b.show()

# # x = torch.randn(8, 1, 64, 64).cuda()
# h = xh[0][:, :, 0, :, :]
# h = torch.cat((xh[0][:, :, 1, :, :], h), dim=1)
# h = torch.cat((xh[0][:, :, 2, :, :], h), dim=1)

# a = h[0, 0, :, :]
# a = transforms.ToPILImage(a)
# trans = transforms.Compose(transforms.Resize((128)))
# a = trans(a)

# b = ToPILImage()(xl[0].detach().cpu())
# b.show()
# y = ifm((xl, xh))



