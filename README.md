# PointNet
This repo is PyTorch implementation for PointNet(https://arxiv.org/abs/1612.00593).

# Environment
Ubuntu 18.04\
Python 3.8.8\
NVIDIA Geforce 2080 Ti\
Cuda 11.1\
PyTorch 1.8.1

# Data
ModelNet-10 dataset (https://modelnet.cs.princeton.edu/) is used.

# File Description
In src/pointnet:

pointnet_trainval.py: python implementation for training and validation\
pointnet_test.py: python implementation for testing\
pointnet_vis.py: python implementation for visualizing the point cloud

# Pre-trained models & Performance
Pre-trained models and performances for models are in "pretrained_models/pointnet/*".

## Table
|         Model        | train_acc | val_acc | test_acc |
|:--------------------:|:---------:|:-------:|:--------:|
|       PointNet       |   95.37   |  94.68  |   92.95  |
| PointNet(w/o trans.) |   94.73   |  94.56  |   89.32  |

## Loss-acc graph
### PointNet
![loss_acc_graph](https://user-images.githubusercontent.com/65886276/201007005-a930e084-8b35-4826-9f7b-a73476175292.png)

### PointNet w/o trans.
![loss_acc_graph](https://user-images.githubusercontent.com/65886276/201007312-ece822a7-ed75-4c96-a6d1-3fb066a9b32a.png)