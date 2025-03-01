import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet18 import ResNet18
import os
import Loss
from torch.utils.data import random_split
import clientTrainer
import copy
from dataprocess import *
from adjacency import *
import json
import numpy as np
from torch.utils.data import DataLoader
from resnet18 import ResNet18
from models import *
# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
device = 'cuda'
# 加载 EMNIST 数据集
emnist_train = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
emnist_test = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
print(len(cifar10))
print(len(mnist_train))
print(len(cifar100))
client_number=10
BATCH_SIZE=128
transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
# 选择标签为a到f的数据

print(len(emnist_train))
# 创建子集