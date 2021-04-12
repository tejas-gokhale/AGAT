import os
from os import listdir
from os.path import isfile, join
import pickle
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SIZE = 128
resize = transforms.Resize([SIZE, SIZE])
resize_32 = transforms.Resize([32, 32])


color_to_label_dict = {'gray': 0, 'red': 1, 'blue': 2, 'green': 3,
					   'brown': 4, 'purple': 5, 'cyan': 6, 'yellow': 7}

eval_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])

class CLEVR(Dataset):
	def __init__(self, root, split='train', attribute=None, domain=None, \
		transform=None, use_preprocessed=False):
		if not attribute:
			with open(os.path.join(root, 'scenes', 'CLEVR_'+split+'_scenes.json'), 'r') as f:
				self.scenes = json.load(f)["scenes"]
		else:
			with open(os.path.join(root, 'scenes', 'CLEVR_'+split+'_'+domain+'_'+attribute+'_scenes.json'), 'r') as f:
				self.scenes = json.load(f)

		# store num_objects as label and img_names to load imgs from
		self.num_objects = []
		self.colors = []
		self.img_names = []
		for s in self.scenes:
			self.num_objects.append(len(s["objects"]))
			self.img_names.append(s["image_filename"])	
			self.colors.append(color_to_label_dict[s["objects"][0]["color"]])	

		self.transform = transform
		self.root = root
		self.split = split
		self.use_preprocessed = use_preprocessed

	def __getitem__(self, index):
		imgfile = self.img_names[index]
		count = self.num_objects[index]
		color = self.colors[index]

		img = Image.open(
				os.path.join(self.root, 'images', 
							 self.split, imgfile)).convert('RGB')
		img = resize(img)

		if self.transform is not None:
			img = self.transform(img)
		else:
			img = eval_transform(img)

		# print(type(color), color, color.shape)
		return img, torch.tensor(color)
	def __len__(self):
		return len(self.img_names)


class CLEVR_aug(Dataset):
	def __init__(self, X_aug, y_aug):	
		self.X_aug = X_aug
		self.y_aug = y_aug

	def __getitem__(self, index):
		img_aug = self.X_aug[index]
		color_aug = self.y_aug[index]

		return img_aug, color_aug
	def __len__(self):
		return len(self.X_aug)


class CLEVR_plus_aug(Dataset):
	def __init__(self, root, X_aug, y_aug, split='train', attribute=None, domain=None, \
		transform=None, use_preprocessed=False):
		if not attribute:
			with open(os.path.join(root, 'scenes', 'CLEVR_'+split+'_scenes.json'), 'r') as f:
				self.scenes = json.load(f)["scenes"]
		else:
			with open(os.path.join(root, 'scenes', 'CLEVR_'+split+'_'+domain+'_'+attribute+'_scenes.json'), 'r') as f:
				self.scenes = json.load(f)

		# store num_objects as label and img_names to load imgs from
		self.num_objects = []
		self.colors = []
		self.img_names = []
		for s in self.scenes:
			self.num_objects.append(len(s["objects"]))
			self.img_names.append(s["image_filename"])	
			self.colors.append(color_to_label_dict[s["objects"][0]["color"]])	

		self.transform = transform
		self.root = root
		self.split = split
		self.use_preprocessed = use_preprocessed

		self.X_aug = X_aug
		self.y_aug = y_aug

	def __getitem__(self, index):
		imgfile = self.img_names[index]
		count = self.num_objects[index]
		color = self.colors[index]

		img = Image.open(
				os.path.join(self.root, 'images', 
							 self.split, imgfile)).convert('RGB')
		img = resize(img)

		if self.transform is not None:
			img = self.transform(img)
		else:
			img = eval_transform(img)

		# imgfile1 = self.img_names[2*index+1]
		# count1 = self.num_objects[2*index+1]
		# color1 = self.colors[2*index+1]

		# img1 = Image.open(
		# 		os.path.join(self.root, 'images', 
		# 					 self.split, imgfile1)).convert('RGB')
		# img1 = resize(img1)

		# if self.transform is not None:
		# 	img1 = self.transform(img1)
		# else:
		# 	img1 = eval_transform(img1)

		img_aug = self.X_aug[index]
		color_aug = self.y_aug[index]

		# print("color", color, "color_aug", color_aug.item())
		# print("type(color)", type(color), "type(color_aug)", type(color_aug.item()))

		img_both = torch.cat([img.unsqueeze(0), img_aug.unsqueeze(0)], dim=0)
		color_both = torch.tensor([color] + [color_aug.item()])

		return img_both, color_both
	def __len__(self):
		return len(self.X_aug)


class MNIST_aug(Dataset):
	def __init__(self, X_aug, y_aug, transform=None):	
		self.X_aug = X_aug
		self.y_aug = y_aug
		self.transform = transform

	def __getitem__(self, index):
		img_aug = self.X_aug[index]
		label_aug = self.y_aug[index]

		if self.transform is not None:
			img_aug = self.transform(img_aug)

		return img_aug, label_aug
	def __len__(self):
		return len(self.X_aug)



class MNIST_plus_aug(Dataset):
	def __init__(self, root, X_aug, y_aug, train=True, transform=None, 
				 target_transform=None, download=False):
		self.processed_folder= 'processed'
		self.training_file = 'training.pt'
		self.test_file = 'test.pt'

		self.transform = transform 
		self.target_transform = target_transform
		
		self.train = train  # training set or test set

		if self.train:
			data_file = self.training_file
		else:
			data_file = self.test_file
		self.data, self.targets = torch.load(
			os.path.join(root, self.processed_folder, data_file))

		self.X_aug = X_aug
		self.y_aug = y_aug

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], int(self.targets[index])

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img.numpy(), mode='L')

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		img_aug = self.X_aug[index]
		label_aug = self.y_aug[index]

		img_both = torch.cat([img.unsqueeze(0), img_aug.unsqueeze(0)], dim=0)

		label_both = torch.tensor([target] + [label_aug.item()])

		return img_both, label_both
		# return img, target

	def __len__(self):
		return len(self.X_aug)


class MNIST_perturbation_sets(Dataset):
	def __init__(self, root, transform=None):	
		onlyfiles = [f for f in listdir(root) if isfile(join(root, f))]
		self.root = root
		self.fnames = [f for f in listdir(root) if isfile(join(root, f))]
		self.transform = transform

	def __getitem__(self, index):
		file = self.fnames[index]
		img = Image.open(os.path.join(self.root, file))
		img = resize_32(img)
		if self.transform is not None:
			img = self.transform(img)
		else:
			img = eval_transform(img)

		label = int(file.split('_')[-1][:-4])

		return img, torch.tensor(label)
	def __len__(self):
		return len(self.fnames)