# Based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3
from models.distortions import GaussianSmoothing
from torch.nn import functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smoothing = GaussianSmoothing(3, 3).to(device)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    # gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
    #                             kernel_size=kernel_size, groups=channels, bias=False)

    # gaussian_filter.weight.data = gaussian_kernel
    # gaussian_filter.weight.requires_grad = False
    
    return gaussian_kernel


class BasicBlock(nn.Module):
	def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.downsample = downsample
		self.stride = stride
		
		self.bn1 = norm_layer(inplanes)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv1 = conv3x3(inplanes, planes, stride)
		
		self.bn2 = norm_layer(planes)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)

	def forward(self, x):
		residual = x 
		residual = self.bn1(residual)
		residual = self.relu1(residual)
		residual = self.conv1(residual)

		residual = self.bn2(residual)
		residual = self.relu2(residual)
		residual = self.conv2(residual)

		if self.downsample is not None:
			x = self.downsample(x)
		return x + residual

class Downsample(nn.Module):
	def __init__(self, nIn, nOut, stride):
		super(Downsample, self).__init__()
		self.avg = nn.AvgPool2d(stride)
		assert nOut % nIn == 0
		self.expand_ratio = nOut // nIn

	def forward(self, x):
		x = self.avg(x)
		return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

class ResNetCifar(nn.Module):
	def __init__(self, depth, width=1, classes=10, channels=3, \
		norm_layer=nn.BatchNorm2d, b=1):
		assert (depth - 2) % 6 == 0         # depth is 6N+2
		self.N = (depth - 2) // 6
		super(ResNetCifar, self).__init__()

		# Following the Wide ResNet convention, we fix the very first convolution
		self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.inplanes = 16
		self.layer1 = self._make_layer(norm_layer, 16 * width)
		self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
		self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
		self.bn = norm_layer(64 * width)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(8)
		self.fc = nn.Linear(64 * width, classes)
		# self.kernel = Variable(
		# 				torch.FloatTensor(
		# 					[[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]))

		## distortions
		# self.sigma = b
		# self.smoothing = GaussianSmoothing(3, 3)


		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				
	def _make_layer(self, norm_layer, planes, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes:
			downsample = Downsample(self.inplanes, planes, stride)
		layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
		self.inplanes = planes
		for i in range(self.N - 1):
			layers.append(BasicBlock(self.inplanes, planes, norm_layer))
		return nn.Sequential(*layers)

	def forward(self, x, return_feat=False, distort=False, b=[1,1]):
		if distort:
			## BLUR
			# kernel = Variable(
			# 			get_gaussian_kernel(3, b[0].item()),
			# 			requires_grad=False
			# 			).to(device)
			# x = F.pad(x, (1, 1, 1, 1), mode='reflect')
			# x = F.conv2d(x, kernel, groups=3)

			## NOISE
			x = x + 0.1*b[1].item() * torch.randn(x.shape).to(device)

			## return augmented image
			x_aug = x

		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.bn(x)
		x = self.relu(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)

		if distort and return_feat:
			return x, self.fc(x), x_aug, b
		elif return_feat:
			return x, self.fc(x)
		elif distort:
			return self.fc(x), x_aug
		else:
			return self.fc(x)
		