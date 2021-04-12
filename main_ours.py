from __future__ import print_function
import argparse
from tqdm import tqdm 
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from skimage.filters import gaussian

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import rotate_batch

from models.distortions import GaussianSmoothing, GaussianLayer
from cat_dataloaders import cat_dataloaders
from dataset import MNIST_aug
from utils.digits_process_dataset import asarray_and_reshape


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
parser.add_argument('--aug_percentage', default=30, type=int)
parser.add_argument('--N_adv', default=10, type=int,
                    help='iterations for adversarial training')
parser.add_argument('--aug_freq', default=5, type=int,
                    help='augment data after every aug_freq epochs')
parser.add_argument('--constraint_coeff', type=float, default=5)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import os
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)

print("Loading data ...")
common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
    'snow', 'frost', 'fog',
    'spatter', 'saturate', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

all_teloaders = []
args.level = 5
for cc in common_corruptions:
    args.corruption = cc
    _, teloader = prepare_test_data(args)
    all_teloaders.append(teloader)

args.corruption = 'original'
_, orig_teloader = prepare_test_data(args)

_, train_loader = prepare_train_data(args)
BATCH_SIZE = args.batch_size
args.batch_size = args.aug_batch_size
_, aug_train_loader = prepare_train_data(args)
args.batch_size = BATCH_SIZE


parameters = list(net.parameters())+list(head.parameters())
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
criterion = nn.CrossEntropyLoss().to(device)
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
l1_loss = nn.L1Loss()

all_err_cls = []
all_err_ssh = []
print('Running...')
print('Error (%)\t\tclean\t\ttest\t\tself-supervised')


if args.resume: 
    checkpoint = torch.load(args.ckpt)
    net.load_state_dict(checkpoint['net'])
    head.load_state_dict(checkpoint['head'])
    ssh.load_state_dict(checkpoint['ssh'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

    net.train()
    ssh.train()

    print('Resuming from', start_epoch, "error_cls", checkpoint['err_cls']) 


all_X_aug, all_y_aug = [], []
FLAG_AUG_START = True
num_steps = (args.nepoch - args.epochs_pre) // args.aug_freq
AUG_STEP = 0
try:
    aug_per_step = args.aug_percentage /num_steps
except:
    aug_per_step = 0

for epoch in range(args.start_epoch, args.nepoch+1):
    

    if epoch < args.epochs_pre:
        net.train()
        ssh.train()
        c_err_cls = []
        avg_err_cls = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # print("inside train")
            optimizer.zero_grad()
            X, label = inputs.to(device), labels.to(device)
            y = net(X)
            loss = criterion(y, label)

            if args.shared is not None:
                inputs_ssh, labels_ssh = rotate_batch(inputs, args.rotation_type)
                inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
                outputs_ssh = ssh(inputs_ssh)
                loss_ssh = criterion(outputs_ssh, labels_ssh)
                loss += loss_ssh

            loss.backward()
            optimizer.step()
        scheduler.step()

    
    else:   
        if epoch % args.aug_freq == 0 or epoch == args.epochs_pre:
            AUG_STEP += 1
            net.eval()
            
            aug_pbar = tqdm(aug_train_loader, ascii=True)

            net.eval()

            b = torch.zeros(1).to(device)
            b_aug = torch.rand(1).to(device)

            for i, (inputs, labels) in enumerate(aug_pbar):
                optimizer.zero_grad()
                X, label = inputs.to(device), labels.to(device)
                X_aug = X.clone().to(device)

                z, y = net(X, return_feat=True)

                aug_optimizer = torch.optim.SGD(
                                    [b_aug.requires_grad_()], 
                                    args.lr_adv )

                b = torch.zeros(2).to(device)
                b_aug = torch.rand(2).to(device)


                for n in range(args.N_adv):
                    aug_optimizer.zero_grad() 

                    z_aug, y_aug, X_aug, b_aug = net(
                                    X, return_feat=True, 
                                    distort=True, b=b_aug)

                    loss_cls = criterion(y_aug, label) + bce_loss(F.softmax(y_aug), F.softmax(y.detach()))
                    loss_const_z = mse_loss(F.sigmoid(z), F.sigmoid(z_aug)) 
                    loss_const_b = mse_loss(b, b_aug) + l1_loss(b, b_aug)
                    loss_const = loss_const_z + loss_const_b.to(device) 
                    loss_adv = loss_cls - args.constraint_coeff*loss_const


                    loss_adv.backward(retain_graph=True)
                    aug_optimizer.step()
                    torch.cuda.empty_cache()

                b_final = b_aug.clone().detach()


                z_aug, y_aug, X_aug, b_aug = net(
                                    X, return_feat=True, 
                                    distort=True, b=b_aug)

                add_X_aug, add_y_aug = asarray_and_reshape(
                                            X_aug.data.cpu().numpy(), 
                                            label.data.cpu().numpy(),
                                            3, 32, 32
                                            )
                if FLAG_AUG_START:
                    all_X_aug = np.copy(add_X_aug)
                    all_y_aug = np.copy(add_y_aug)
                    FLAG_AUG_START = False
                else:
                    all_X_aug = np.concatenate([all_X_aug, add_X_aug])
                    all_y_aug = np.concatenate([all_y_aug, add_y_aug])


                aug_pbar.set_description(
                    'augmenting iter: {}; LR = {}'.format(i, args.lr_adv))

                if i == int(len(aug_train_loader) * aug_per_step/100):
                    print("b_aug", b_final)
                    print("torch.sum(X - X_aug)", torch.sum(X - X_aug))
                    with torch.no_grad():
                        samples_aug = vutils.make_grid(
                                        X_aug, nrow=8, normalize=True)
                        vutils.save_image(samples_aug, 'outs/samples_aug.jpg')

                        samples_X = vutils.make_grid(
                                        X, nrow=8, normalize=True)
                        vutils.save_image(samples_X, 'outs/samples_orig.jpg')

                    break
                                                  
        all_X_aug = torch.tensor(all_X_aug)
        all_y_aug = torch.tensor(all_y_aug)

        # print("TRAINING ON SRC+AUG. Epoch:", epoch, "")
        aug_train_dset = MNIST_aug(all_X_aug, all_y_aug)
        aug_size = int(args.batch_size * aug_per_step /100)
        new_aug_train_loader = DataLoader(
                                aug_train_dset, batch_size=aug_size,
                                shuffle=True, num_workers=1, drop_last=True
                                )
        new_train_size = args.batch_size - aug_size

        both_loader = cat_dataloaders([train_loader, new_aug_train_loader])
        print(
                "TRAINING ON SRC+AUG. Epoch", epoch, 
                "-added samples:", all_X_aug.shape[0], 
                "total batches", len(both_loader)
                )
        sys.stdout.flush()



        net.train()
        ssh.train()
        c_err_cls = []
        avg_err_cls = 0

        for i, (inputs, labels) in enumerate(both_loader):
            optimizer.zero_grad()
            X, label = inputs.to(device), labels.to(device)
            y = net(X)
            loss = criterion(y, label)

            if args.shared is not None:
                inputs_ssh, labels_ssh = rotate_batch(inputs, args.rotation_type)
                inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
                outputs_ssh = ssh(inputs_ssh)
                loss_ssh = criterion(outputs_ssh, labels_ssh)
                loss += loss_ssh

            loss.backward()
            optimizer.step()
        scheduler.step()


    if epoch % args.val_freq == 1 or epoch == args.nepoch:
        for i in range(len(all_teloaders)):
            c_err_cls.append(round(test(all_teloaders[i], net)[0], 4))
        orig_err_cls = test(orig_teloader, net)[0]
        err_cls = sum(c_err_cls)/len(c_err_cls)
        all_err_cls.append(err_cls)

        err_ssh = 0 if args.shared is None else test(teloader, ssh, sslabel='expand')[0]
        all_err_ssh.append(err_ssh)

        print((
            'Epoch %d/%d:' %(epoch, args.nepoch)).ljust(24) +
            '%.2f\t\t%.2f\t\t%.2f' %(orig_err_cls*100, err_cls*100, err_ssh*100))
        print(common_corruptions)
        print(c_err_cls)
        torch.save((all_err_cls, all_err_ssh), args.outf + '/loss_aug.pth')
        plot_epochs(all_err_cls, all_err_ssh, args.outf + '/loss_aug.pdf')

        state = {
                    'err_cls': err_cls, 
                    'err_ssh': err_ssh,
                    'net': net.state_dict(), 
                    'head': head.state_dict(),
                    'ssh': ssh.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'loss': loss, 
                    'epoch': epoch
                }
        torch.save(state, args.outf + '/ckpt_aug_' + str(epoch) + '.pth')
        sys.stdout.flush()
