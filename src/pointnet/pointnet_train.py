import os
import sys
import torch
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('../..')
from models.pointnet import PointNet, pointnet_loss
from models.pointnet_fxia22 import PointNetCls
from models.pointnet_wo_transform import PointNet_wo_transform
from datasets.modelnet import ModelNetDataset

data_root = '/share/ModelNet10'
save_dir = '/share/DLL/project/pretrained_models/pointnet'
data_ver = '10'

batch_size = 10
num_epoch = 250
learning_rate = 1e-3


def save_model(model, path, option='overall'):	# .pt, .pth, .pkl
	##### OVERALL MODEL STATE #####
	if option == 'overall':
		torch.save(model, path)

	##### ONLY MODEL PARAMETERS #####
	elif option == 'params':
		torch.save(model.state_dict(), path)

	##### ONLY MODEL PARAMETERS #####
	else:
		raise ValueError('Invalid input of option in function save_model(). Cannot save model.')


def main():
	dataset = ModelNetDataset(root=data_root, name=data_ver, split='train')
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

	pointnet = PointNet(k=10).to(device)
	# pointnet = PointNetCls(k=10).to(device)
	# pointnet = PointNet_wo_transform(k=10).to(device)

	pointnet.train()

	optimizer = torch.optim.Adam(pointnet.parameters(), lr=learning_rate)
	# criterion = nn.CrossEntropyLoss().to(device)
	criterion = pointnet_loss()

	loss_list = []
	accuracy_list = []
	for epoch in tqdm(range(num_epoch)):
		correct = 0
		total = 0
		epoch_loss = 0
		# for batch_idx, samples in enumerate(tqdm(dataloader)):
		for batch_idx, samples in enumerate(dataloader):
			optimizer.zero_grad()

			x_train, y_train = samples
			x_train = x_train.to(device)

			pred, in_trans, local_feat = pointnet(x_train)
			y_train = torch.squeeze(y_train)
			y_train = np.array(y_train, dtype=float)
			y_train = torch.LongTensor(y_train)
			y_train = y_train.to(device)

			_, pred_label = torch.max(pred, 1)
			correct += (pred_label == y_train).sum().item()
			total += len(y_train)

			loss = criterion(pred, y_train, local_feat)

			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()

		epoch_loss /= len(dataloader)
		accuracy = correct / total * 100

		loss_list.append(epoch_loss)
		accuracy_list.append(accuracy)

		# print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, epoch_loss, accuracy))

		if epoch % 10 == 0:
			print('\nEpoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, epoch_loss, accuracy))

			save_file_name = pointnet.name + '_epoch' + str(epoch) + '.pth'
			save_path = os.path.join(save_dir, save_file_name)

			print('Saving model to ', save_path, "...")
			save_model(pointnet, save_path)
			print('Done.')

			list_file_name = 'loss_list_epoch' + str(epoch) + '.pkl'
			list_file_path = os.path.join(save_dir, list_file_name)
			accuracy_file_name = 'accuracy_epoch' + str(epoch) + '.pkl'
			accuracy_file_path = os.path.join(save_dir, accuracy_file_name)
			with open(list_file_path, "wb") as ll:
				pickle.dump(loss_list, ll)
			with open(accuracy_file_path, "wb") as al:
				pickle.dump(accuracy_list, al)


	print('\nEpoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, epoch_loss, accuracy))

	save_file_name = pointnet.name + '_latest.pth'
	save_path = os.path.join(save_dir, save_file_name)

	print('Saving model to ', save_path, "...")
	save_model(pointnet, save_path)
	print('Done.')

	list_file_name = 'loss_list.pkl'
	list_file_path = os.path.join(save_dir, list_file_name)
	accuracy_file_name = 'accuracy_list.pkl'
	accuracy_file_path = os.path.join(save_dir, accuracy_file_name)
	with open(list_file_path, "wb") as ll:
		pickle.dump(loss_list, ll)
	with open(accuracy_file_path, "wb") as al:
		pickle.dump(accuracy_list, al)


if __name__ == '__main__':
	# np.random.seed(0)
	torch.manual_seed(1)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == 'cuda':
		torch.cuda.manual_seed_all(1)
	print(device + " is available")

	main()

	# print(“Allocated:”, round(torch.cuda.memory_allocated(0)/1024**3,1), “GB”)