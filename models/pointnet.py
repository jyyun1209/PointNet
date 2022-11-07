import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# Input Transform (IN: [B,3,N], OUT: [B,3,3])
class in_trans(nn.Module):
	def __init__(self):
		super(in_trans, self).__init__()
		self.conv1 = nn.Conv1d(3, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)

		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)

		self.relu = nn.ReLU()
		
		self.weights = nn.Linear(256, 3*3, bias=True)
		torch.nn.init.zeros_(self.weights.weight.data)
		self.weights.bias.data = torch.from_numpy(np.identity(3).flatten().astype(np.float32))

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		x, _ = torch.max(x, 2, keepdim=True)	# [B,1024,1]
		x = x.view(-1, 1024)

		x = self.fc1(x)
		x = self.bn4(x)
		x = self.relu(x)

		x = self.fc2(x)
		x = self.bn5(x)
		x = self.relu(x)	# [B, 256]

		# weights = Variable(torch.from_numpy(np.zeros(9).astype(np.float32)).repeat(256,1)).view(1, 256, 9).repeat(x.shape[0], 1, 1)
		# bias = Variable(torch.from_numpy(np.identity(3).flatten().astype(np.float32))).repeat(x.shape[0], 1)

		# if x.is_cuda:
			# self.weights = self.weights.cuda()
			# bias = bias.cuda()

		# x = x[:, None, :]

		# print(x.shape)
		# print(self.weights)
		# x = torch.matmul(x, self.weights)	# [B, 9]
		# x = torch.squeeze(x)
		x = self.weights(x)

		# x = x + bias
		x = x.view(-1, 3, 3)

		return x


# Shared MLP 1 (IN: [B,3,N], OUT: [B,64,N])
class s_mlp_1(nn.Module):
	def __init__(self):
		super(s_mlp_1, self).__init__()
		self.conv1 = nn.Conv1d(3, 64, 1)
		self.conv2 = nn.Conv1d(64, 64, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)

		self.relu = nn.ReLU()
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		return x
		

# Feature Transform
class feat_trans(nn.Module):
	def __init__(self):
		super(feat_trans, self).__init__()
		self.conv1 = nn.Conv1d(64, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)

		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)

		self.relu = nn.ReLU()

		self.weights = nn.Linear(256, 64*64, bias=True)
		torch.nn.init.zeros_(self.weights.weight.data)
		self.weights.bias.data = torch.from_numpy(np.identity(64).flatten().astype(np.float32))

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		x, _ = torch.max(x, 2, keepdim=True)	# [B,1024,1]
		x = x.view(-1, 1024)

		x = self.fc1(x)
		x = self.bn4(x)
		x = self.relu(x)

		x = self.fc2(x)
		x = self.bn5(x)
		x = self.relu(x)	# [B, 256]

		# weights = Variable(torch.from_numpy(np.zeros(64*64).astype(np.float32)).repeat(256,1)).view(1, 256, 64*64).repeat(x.shape[0], 1, 1)
		# bias = Variable(torch.from_numpy(np.identity(64).flatten().astype(np.float32))).repeat(x.shape[0], 1)

		# if x.is_cuda:
		# 	weights = weights.cuda()
		# 	bias = bias.cuda()

		# print(weights)

		# x = x[:, None, :]
		# x = torch.matmul(x, weights)	# [B, 9]
		# x = torch.squeeze(x)

		# x = x + bias

		x = self.weights(x)

		x = x.view(-1, 64, 64)

		return x

# Shared MLP 2 (64, 128, 1024)
class s_mlp_2(nn.Module):
	def __init__(self):
		super(s_mlp_2, self).__init__()
		self.conv1 = nn.Conv1d(64, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		return x

# Output MLP (512, 256, k)
class out_mlp(nn.Module):
	def __init__(self, k=10):
		super(out_mlp, self).__init__()
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, k)

		self.dropout = nn.Dropout(p=0.3)

		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.dropout(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.fc2(x)
		x = self.dropout(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.fc3(x)

		return x

# Overall Structure of PointNet
class PointNet(nn.Module):
	def __init__(self, k=10):
		super(PointNet, self).__init__()
		self.name = 'PointNet'
		self.input_transform = in_trans()
		self.shared_mlp_1 = s_mlp_1()
		self.feature_transform = feat_trans()
		self.shared_mlp_2 = s_mlp_2()
		self.out_mlp = out_mlp(k=10)

		# self.criterion = nn.CrossEntropyLoss()

	def forward(self, x):								# input: [B, 3, N]
		# n_pts = x.shape[2]
		in_trans = self.input_transform(x)				# input_trans_mat: [B, 3, 3]
		# print(in_trans.shape)

		x = torch.transpose(x, 2, 1)
		x = torch.bmm(x, in_trans)	
		x = torch.transpose(x, 2, 1)					# transformed_input: [B, 3, N]
		transformed_input = x

		x = self.shared_mlp_1(x)						# after_mlp1: [B, 64, N]		

		feat_trans = self.feature_transform(x)			# feat_trans_mat: [B, 64, 64]

		x = torch.transpose(x, 2, 1)
		x = torch.bmm(x, feat_trans)
		local_feat = torch.transpose(x, 2, 1)			# local_feature: [B, 64, N]

		x = self.shared_mlp_2(local_feat)				# after_mlp2: [B, 1024, N]

		global_feat, _ = torch.max(x, 2, keepdim=True)
		global_feat = torch.squeeze(global_feat, 2)		# global_feature: [B, 1024]
		
		x = self.out_mlp(global_feat)					# after_out_mlp: [B, 10]

		return x, transformed_input, feat_trans

class pointnet_loss(nn.Module):
	def __init__(self):
		super(pointnet_loss, self).__init__()


	def forward(self, pred, label, feat_trans):
		loss = F.cross_entropy(pred, label)

		d = feat_trans.size()[1]
		I = torch.eye(d)[None, :, :]
		if feat_trans.is_cuda:
			I = I.cuda()

		sub = I - torch.bmm(feat_trans, feat_trans.transpose(2, 1))
		mat_diff_loss = torch.mean(torch.sqrt(sub.pow(2).sum(dim=(1,2)) + 1e-8))
		# mat_diff_loss = torch.mean(torch.norm(I - torch.bmm(feat_trans, feat_trans.transpose(2, 1)), dim=(1, 2)))
		# mat_diff_loss = torch.mean(torch.sqrt(sub.pow(2).sum(dim=(1, 2))))
		# mat_diff_loss = torch.mean(sub**2)

		return loss + mat_diff_loss * 0.001