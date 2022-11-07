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
from models.pointnet import PointNet
from datasets.modelnet import ModelNetDataset

data_root = '/share/ModelNet10'
model_path = '/share/DLL/project/pretrained_models/pointnet/PointNet_best.pth'
data_ver = '10'


def vis_3d(pcd, show_origin=False, origin_size=3, show_grid=False):
    cloud = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    
    if isinstance(pcd, type(cloud)):
        pass
    elif isinstance(pcd, np.ndarray):
        cloud.points = v3d(pcd)
        
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))
    
    # set front, lookat, up, zoom to change initial view
    o3d.visualization.draw_geometries([cloud, coord])


def main():
	dataset = ModelNetDataset(root=data_root, name=data_ver, split='test')
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
	# print(len(dataloader))
	pointnet = torch.load(model_path, map_location=device)

	pointnet.eval()
	with torch.no_grad():
		correct = 0
		for batch_idx, samples in enumerate(dataloader):
			x_test_raw, y_test = samples

			x_test = x_test_raw
			x_test = x_test.to(device)

			pred, transformed_input, _ = pointnet(x_test)
			y_test = torch.squeeze(y_test, 0)
			y_test = np.array(y_test, dtype=np.float)
			y_test = torch.LongTensor(y_test)
			y_test = y_test.to(device)

			_, pred_label = torch.max(pred, 1)
			correct += (pred_label == y_test).sum().item()

			print(dataset.classes[pred_label[0]], dataset.classes[y_test[0]])

			vis_pts = torch.squeeze(x_test_raw).T
			vis_pts = np.array(vis_pts)
			print(vis_pts.shape)
			vis_3d(vis_pts)

			vis_pts = torch.squeeze(transformed_input.cpu()).T
			vis_pts = np.array(vis_pts)
			print(vis_pts.shape)
			vis_3d(vis_pts)

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