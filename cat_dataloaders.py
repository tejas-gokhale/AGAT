import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset import CLEVR, CLEVR_aug

transform = transforms.Compose([
        transforms.Pad(8),
        transforms.RandomCrop([128, 128]),
        transforms.RandomRotation(2.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# adapted from the post on 
# https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/35

class cat_dataloaders():
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders, proportions=[1,1]):
        self.dataloaders = dataloaders
        self.proportions = proportions        

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        ## this is more generic -- for n dataloaders
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter))

        ## we only care about 2 datasets
        img1, label1 = next(self.loader_iter[0])
        img2, label2 = next(self.loader_iter[1])

        # print(label1, label2)

        img_both = torch.cat([img1, img2], dim=0)
        label_both = torch.cat([label1, label2], dim=0)

        return img_both, label_both

    def __len__(self):
        L = 0
        for i, dd in enumerate(self.dataloaders):
            L += int(self.proportions[i] * len(dd))

        return L



class DEBUG_dataset(Dataset):
    def __init__(self,alpha):
        self.d = (torch.arange(20) + 1) * alpha
    def __len__(self):
        return self.d.shape[0]
    def __getitem__(self, index):
        return self.d[index]

# train_dl1 = DataLoader(DEBUG_dataset(10), batch_size = 4,num_workers = 0 , shuffle=True)
# train_dl2 = DataLoader(DEBUG_dataset(1), batch_size = 4,num_workers = 0 , shuffle=True)



# train_dset = CLEVR(
#                 root='/home/tgokhale/work/data/clevr/CLEVR_singles/',
#                 split='train', 
#                 transform=transform
#                 )
# train_loader = DataLoader(
#                     train_dset, batch_size=64 - int( 64*15/100), 
#                     shuffle=True, num_workers=1, drop_last=True
#                     )
# X_aug = torch.rand(1000, 3, 128, 128)
# y_aug = torch.randint(low=0, high=7, size=[1000])

# print(X_aug.shape, y_aug.shape)


# aug_train_dset = CLEVR_aug(X_aug, y_aug)

# aug_train_loader = DataLoader(
#                     aug_train_dset, batch_size=int(64*15/100),
#                     shuffle=True, num_workers=1, drop_last=True
#                     )


# tmp = cat_dataloaders([train_loader, aug_train_loader])

# for i, (img, label) in enumerate(tmp):
#     print(img.shape, label.shape)
#     break