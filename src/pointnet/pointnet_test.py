import os
import sys
import torch
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('../..')
from datasets.modelnet import ModelNetDataset

data_root = '/share/ModelNet10'
model_path = '/share/DLL/project/pretrained_models/pointnet/pointnet_final/PointNet_best.pth'
# model_path = '/share/DLL/project/pretrained_models/pointnet/PointNet_best.pth'
data_ver = '10'


def main():
	dataset = ModelNetDataset(root=data_root, name=data_ver, split='test')
	# print(dataset.classes)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
	# print(len(dataloader))
	pointnet = torch.load(model_path, map_location=device)

	pointnet.eval()
	with torch.no_grad():
		correct = 0
		for batch_idx, samples in enumerate(tqdm(dataloader)):
			x_test, y_test = samples
			x_test = x_test.to(device)

			pred, _, _ = pointnet(x_test)
			y_test = torch.squeeze(y_test, 0)
			y_test = np.array(y_test, dtype=float)
			y_test = torch.LongTensor(y_test)
			y_test = y_test.to(device)

			_, pred_label = torch.max(pred, 1)
			# print(pred_label, y_test)
			correct += (pred_label == y_test).sum().item()

		accuracy = correct / len(dataloader) * 100

		print('\nTesting Accuracy {:2.2f}%'.format(accuracy))


if __name__ == '__main__':
	# np.random.seed(0)
	torch.manual_seed(1)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == 'cuda':
		torch.cuda.manual_seed_all(1)
	print(device + " is available")

	main()
