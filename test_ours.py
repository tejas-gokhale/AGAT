from __future__ import print_function
import argparse
from tqdm import tqdm 
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from skimage.filters import gaussian

# from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
# from utils.rotation import rotate_batch

# from models.distortions import GaussianSmoothing, GaussianLayer
# from cat_dataloaders import cat_dataloaders
# from dataset import MNIST_aug
# from utils.digits_process_dataset import asarray_and_reshape


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='../../data')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--aug_batch_size', default=32, type=int)
parser.add_argument('--group_norm', default=0, type=int)
########################################################################
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr_adv', default=0.0001, type=float)
parser.add_argument('--nepoch', default=75, type=int)
parser.add_argument('--start_epoch', default=99, type=int)
parser.add_argument('--epochs_pre', default=101, type=int)
parser.add_argument('--milestone_1', default=50, type=int)
parser.add_argument('--milestone_2', default=65, type=int)
parser.add_argument('--rotation_type', default='rand')
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--corruption', default='shot_noise')
parser.add_argument('--val_freq', type=int, default=5)
parser.add_argument('--resume', action="store_true", default=False)
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--aug_percentage', default=50, type=int)
parser.add_argument('--N_adv', default=10, type=int,
                    help='iterations for adversarial training')
parser.add_argument('--aug_freq', default=5, type=int,
                    help='augment data after every aug_freq epochs')
parser.add_argument('--constraint_coeff', type=float, default=5)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cudnn.benchmark = True
net, ext, head, ssh = build_model(args)

common_corruptions = [
    'fog', 'snow', 'frost', 
    'zoom_blur', 'defocus_blur', 'glass_blur', 'motion_blur',
    'shot_noise', 'impulse_noise', 'gaussian_noise',   
    'jpeg_compression', 'pixelate', 'elastic_transform', 'brightness', 'contrast'  
    ]


# print('Running...')
# print('Error (%)\t\ttest')

parameters = list(net.parameters())+list(head.parameters())
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)


args.level = 5
args.corruption = 'original'
_, teloader = prepare_test_data(args)

for i, X in enumerate(teloader):
    # print(i, X[0].shape, X[1].shape)
    samples = vutils.make_grid(X[0], nrow=8, normalize=True, range=(-1., 1.))
    vutils.save_image(samples, './c10c_original.jpg', nrow=1)


    break

args.level = 5
args.corruption = 'shot_noise'
_, teloader = prepare_test_data(args)

for i, X in enumerate(teloader):
    # print(i, X[0].shape, X[1].shape)
    samples = vutils.make_grid(X[0], nrow=8, normalize=True, range=(-1., 1.))
    vutils.save_image(samples, './c10c_shotnoise.jpg', nrow=1)


    break


args.level = 5
args.corruption = 'pixelate'
_, teloader = prepare_test_data(args)

for i, X in enumerate(teloader):
    # print(i, X[0].shape, X[1].shape)
    samples = vutils.make_grid(X[0], nrow=8, normalize=True, range=(-1., 1.))
    vutils.save_image(samples, './c10c_pixelate.jpg', nrow=1)


    break

args.level = 5
args.corruption = 'frost'
_, teloader = prepare_test_data(args)

for i, X in enumerate(teloader):
    # print(i, X[0].shape, X[1].shape)
    samples = vutils.make_grid(X[0], nrow=8, normalize=True, range=(-1., 1.))
    vutils.save_image(samples, './c10c_frost.jpg', nrow=1)


    break


'''
if args.resume: 
    checkpoint = torch.load(args.ckpt)
    net.load_state_dict(checkpoint['net'])
    # head.load_state_dict(checkpoint['head'])
    # ssh.load_state_dict(checkpoint['ssh'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

    net.eval()
    ssh.eval()

for LEVEL in range(1, 6):

    all_teloaders = []
    args.level = LEVEL
    for cc in common_corruptions:
        args.corruption = cc
        _, teloader = prepare_test_data(args)
        all_teloaders.append(teloader)

    all_err_cls = []
    all_err_ssh = []
    c_err_cls = []
    avg_err_cls = 0

    print("LEVEL", LEVEL)
    for i in range(len(all_teloaders)):
        c_err_cls.append(round(test(all_teloaders[i], net)[0], 4))
    # orig_err_cls = test(orig_teloader, net)[0]
    err_cls = sum(c_err_cls)/len(c_err_cls)
    all_err_cls.append(err_cls)

    # err_ssh = 0 if args.shared is None else test(teloader, ssh, sslabel='expand')[0]
    # all_err_ssh.append(err_ssh)

    print("AVG---", (1-err_cls)*100)
    # print(common_corruptions)
    print(c_err_cls)
    print("weather", 100*(1- sum(c_err_cls[:3])/3))
    print("blur", 100*(1- sum(c_err_cls[3:7])/4))
    print("noise", 100*(1- sum(c_err_cls[7:10])/3))
    print("digital", 100*(1- sum(c_err_cls[10:])/5))
    print("--------------------------------------------------------------")


'''