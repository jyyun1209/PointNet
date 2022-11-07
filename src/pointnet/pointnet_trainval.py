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
num_epoch = 25
learning_rate = 1e-3

# torch.autograd.set_detect_anomaly(True)

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
	train_dataset = ModelNetDataset(root=data_root, name=data_ver, split='train')
	train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

	val_dataset = ModelNetDataset(root=data_root, name=data_ver, split='val')
	val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)

	pointnet = PointNet(k=10).to(device)
	# pointnet = PointNetCls(k=10).to(device)
	# pointnet = PointNet_wo_transform(k=10).to(device)

	optimizer = torch.optim.Adam(pointnet.parameters(), lr=learning_rate)
	# criterion = nn.CrossEntropyLoss().to(device)	# for wo. trans. ver.
	criterion = pointnet_loss()					# for my and fxia22 ver.

	train_loss_list = []
	train_accuracy_list = []
	val_loss_list = []
	val_accuracy_list = []
	best_acc = 0
	for epoch in tqdm(range(num_epoch)):
		pointnet.train()

		correct = 0
		total = 0
		epoch_loss = 0
		for batch_idx, samples in enumerate(train_dataloader):
			optimizer.zero_grad()

			x_train, y_train = samples
			x_train = x_train.to(device)

			pred, _, local_feat = pointnet(x_train)
			y_train = torch.squeeze(y_train)
			y_train = np.array(y_train, dtype=float)
			y_train = torch.LongTensor(y_train)
			y_train = y_train.to(device)

			_, pred_label = torch.max(pred, 1)
			correct += (pred_label == y_train).sum().item()
			total += len(y_train)

			# loss = criterion(pred, y_train)					# for wo. trans. ver.
			loss = criterion(pred, y_train, local_feat)	# for my and fxia22 ver.

			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()

		epoch_loss /= len(train_dataloader)
		accuracy = correct / total * 100

		train_loss_list.append(epoch_loss)
		train_accuracy_list.append(accuracy)

		pointnet.eval()
		with torch.no_grad():
			for batch_idx, samples in enumerate(val_dataloader):
				x_val, y_val = samples
				x_val = x_val.to(device)

				pred, _, local_feat = pointnet(x_val)
				y_val = torch.squeeze(y_val)
				y_val = np.array(y_val, dtype=float)
				y_val = torch.LongTensor(y_val)
				y_val = y_val.to(device)

				_, pred_label = torch.max(pred, 1)
				correct += (pred_label == y_val).sum().item()
				total += len(y_val)

				# loss = criterion(pred, y_val)					# for wo. trans. ver.
				loss = criterion(pred, y_val, local_feat)		# for my and fxia22 ver.
				epoch_loss += loss.item()

			epoch_loss /= len(val_dataloader)
			accuracy = correct / total * 100

			val_loss_list.append(epoch_loss)
			val_accuracy_list.append(accuracy)

			# LATEST
			save_file_name = pointnet.name + '_latest.pth'
			save_path = os.path.join(save_dir, save_file_name)

			# print('Saving model to ', save_path, "...")
			save_model(pointnet, save_path)
			# print('Done.')

			train_list_file_name = 'train_loss_list_latest.pkl'
			train_list_file_path = os.path.join(save_dir, train_list_file_name)
			train_accuracy_file_name = 'train_acc_list_latest.pkl'
			train_accuracy_file_path = os.path.join(save_dir, train_accuracy_file_name)
			val_list_file_name = 'val_loss_list_latest.pkl'
			val_list_file_path = os.path.join(save_dir, val_list_file_name)
			val_accuracy_file_name = 'val_acc_list_latest.pkl'
			val_accuracy_file_path = os.path.join(save_dir, val_accuracy_file_name)
			with open(train_list_file_path, "wb") as tll:
				pickle.dump(train_loss_list, tll)
			with open(train_accuracy_file_path, "wb") as tal:
				pickle.dump(train_accuracy_list, tal)
			with open(val_list_file_path, "wb") as vll:
				pickle.dump(val_loss_list, vll)
			with open(val_accuracy_file_path, "wb") as val:
				pickle.dump(val_accuracy_list, val)

			# BEST
			if best_acc <= val_accuracy_list[-1]:
				best_acc = val_accuracy_list[-1]
				save_file_name = pointnet.name + '_best.pth'
				save_path = os.path.join(save_dir, save_file_name)

				print('\n[TRAIN] Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, train_loss_list[-1], train_accuracy_list[-1]))
				print('\n[VALID] Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, val_loss_list[-1], val_accuracy_list[-1]))

				# print('Saving model to ', save_path, "...")
				save_model(pointnet, save_path)
				# print('Done.')

				train_list_file_name = 'train_loss_list_best.pkl'
				train_list_file_path = os.path.join(save_dir, train_list_file_name)
				train_accuracy_file_name = 'train_acc_list_best.pkl'
				train_accuracy_file_path = os.path.join(save_dir, train_accuracy_file_name)
				val_list_file_name = 'val_loss_list_best.pkl'
				val_list_file_path = os.path.join(save_dir, val_list_file_name)
				val_accuracy_file_name = 'val_acc_list_best.pkl'
				val_accuracy_file_path = os.path.join(save_dir, val_accuracy_file_name)
				with open(train_list_file_path, "wb") as tll:
					pickle.dump(train_loss_list, tll)
				with open(train_accuracy_file_path, "wb") as tal:
					pickle.dump(train_accuracy_list, tal)
				with open(val_list_file_path, "wb") as vll:
					pickle.dump(val_loss_list, vll)
				with open(val_accuracy_file_path, "wb") as val:
					pickle.dump(val_accuracy_list, val)

			else:
				if epoch % 10 == 0:
					print('\n[TRAIN] Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, train_loss_list[-1], train_accuracy_list[-1]))
					print('\n[VALID] Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, val_loss_list[-1], val_accuracy_list[-1]))

	print('\n[TRAIN] Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, train_loss_list[-1], train_accuracy_list[-1]))
	print('\n[VALID] Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch+1, num_epoch, val_loss_list[-1], val_accuracy_list[-1]))


if __name__ == '__main__':
	# np.random.seed(0)
	torch.manual_seed(1)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == 'cuda':
		torch.cuda.manual_seed_all(1)
	print(device + " is available")

	main()

	# print(“Allocated:”, round(torch.cuda.memory_allocated(0)/1024**3,1), “GB”)