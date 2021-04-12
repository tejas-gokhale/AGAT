import math
import numbers
import torch
import numpy as np 
from torch import nn
from torch.nn import functional as F
import scipy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, dim=2):
        super(GaussianSmoothing, self).__init__()
        # if isinstance(kernel_size, numbers.Number):
        #     kernel_size = [kernel_size] * dim
        # if isinstance(sigma, numbers.Number):
        #     sigma = [sigma] * dim

        # # The gaussian kernel is the product of the
        # # gaussian function of each dimension.
        # kernel = 1
        # meshgrids = torch.meshgrid(
        #     [
        #         torch.arange(size, dtype=torch.float32)
        #         for size in kernel_size
        #     ]
        # )
        # for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        #     mean = (size - 1) / 2
        #     kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
        #               torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # # Make sure sum of values in gaussian kernel equals 1.
        # kernel = kernel / torch.sum(kernel)

        # # Reshape to depthwise convolutional weight
        # kernel = kernel.view(1, 1, *kernel.size())
        # kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # self.register_buffer('weight', kernel)
        # self.groups = channels

        # if dim == 1:
        #     self.conv = F.conv1d
        # elif dim == 2:
        #     self.conv = F.conv2d
        # elif dim == 3:
        #     self.conv = F.conv3d
        # else:
        #     raise RuntimeError(
        #         'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
        #     )

        self.channels = channels
        self.kernel_size = kernel_size 
        if isinstance(kernel_size, numbers.Number):
            self.kernel_size = [kernel_size] * dim
        self.dim = dim

    def forward(self, input, sigma):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * self.dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in self.kernel_size
            ]
        )
        for size, std, mgrid in zip(self.kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = self.channels

        if self.dim == 1:
            self.conv = F.conv1d
        elif self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        return self.conv(input, weight=self.weight, groups=self.groups)


# smoothing = GaussianSmoothing(3, 5, 1)
# input = torch.rand(1, 3, 100, 100)
# input = F.pad(input, (2, 2, 2, 2), mode='reflect')
# output = smoothing(input)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean



class GaussianLayer(nn.Module):
    def __init__(self, sigma):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(2), 
            nn.Conv2d(3, 3, 5, stride=1, padding=0, bias=None, groups=3)
        )

        self.weights_init(sigma).to(device)
    def forward(self, x):
        return self.seq(x)

    def weights_init(self, sigma):
        n= np.zeros((5,5))
        n[2,2] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


# x = torch.rand(3, 1, 5, 5)
# s = GaussianSmoothing(1, 3)
# x_s = s(x, 1)

# print(x)
# print(x_s)