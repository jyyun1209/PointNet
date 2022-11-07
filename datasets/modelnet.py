import os
import glob
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

import numpy as np
import open3d as o3d

import time

class ModelNetDataset(Dataset):
	def __init__(self, root, name, split='train', num_points=2500):
		self.root = root
		self.name = name
		self.split = split
		self.num_points = num_points

		if self.split == 'test':
			self.raw_dirs = os.path.join(root, 'raw')
			self.classes = [directory for directory in os.listdir(self.raw_dirs) if os.path.isdir(os.path.join(self.raw_dirs, directory))]
			self.classes.sort()

			self.dataset_files = []
			self.labels_list = []

			for label, _dir in enumerate(self.classes):
				dirpath = os.path.join(self.raw_dirs, _dir + '/' + self.split)
				filelist = glob.glob(os.path.join(dirpath, '*'))
				self.dataset_files += filelist
				self.labels_list += [label] * len(filelist)
		else:
			self.raw_dirs = os.path.join(root, 'raw')
			self.classes = [directory for directory in os.listdir(self.raw_dirs) if os.path.isdir(os.path.join(self.raw_dirs, directory))]
			self.classes.sort()
			
			self.dataset_files = []
			self.labels_list = []

			for label, _dir in enumerate(self.classes):
				dirpath = os.path.join(self.raw_dirs, _dir + '/train')
				filelist = glob.glob(os.path.join(dirpath, '*'))

				tot_file_num = len(filelist)
				train_file_num = int(tot_file_num * 0.9)
				if self.split == 'train':
					self.dataset_files += filelist[:train_file_num]
					self.labels_list += [label] * len(filelist[:train_file_num])
				else:
					self.dataset_files += filelist[train_file_num:]
					self.labels_list += [label] * len(filelist[train_file_num:])

	def __len__(self):
		return len(self.labels_list)

	def __getitem__(self, idx):
		filepath = self.dataset_files[idx]
		mesh = o3d.io.read_triangle_mesh(filepath)
		pcd = mesh.sample_points_uniformly(number_of_points=self.num_points)
		pcd = np.asarray(pcd.points, dtype=np.float32)

		pcd = pcd - np.expand_dims(np.mean(pcd, axis=0), 0)
		dist = np.max(np.sqrt(np.sum(pcd**2, axis=1)), 0)
		pcd = pcd / dist
		pcd = pcd.T

		pcd = torch.from_numpy(pcd)
		pcd = torch.FloatTensor(pcd)

		return pcd, torch.tensor([self.labels_list[idx]])
