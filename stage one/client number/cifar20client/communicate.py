import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import *
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
import numpy
dirichlet_alpha = 0.5
torch.manual_seed(42)
numpy.random.seed(42)
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# 超参数设置
# 准备数据集并预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
device = 'cuda'
# 加载 EMNIST 数据集
device = 'cuda'
# 加载 EMNIST 数据集
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
publicdata,clientdatas = random_split(cifar10_train,[2000,len(cifar10_train)-2000])
print(len(publicdata))
print(len(clientdatas))
client_number=20
BATCH_SIZE=128
transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
client_data_size = len(clientdatas) // client_number
client_datasets = random_split(clientdatas, [client_data_size] * client_number)
# client_datasets = get_dataset(clientdatas, client_number, dirichlet_alpha)
for i in range(0, client_number):
    print(len(client_datasets[i]))
# 创建 DataLoader
client_loaders = [torch.utils.data.DataLoader(dataset, BATCH_SIZE,shuffle=True) for dataset in client_datasets]
test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=100, shuffle=True)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
public_loader = torch.utils.data.DataLoader(publicdata, batch_size=128, shuffle=False)
num_rounds= 20
# 定义损失函数和优化方式
def main():
    round_idx = 0
    num_epochs=5
    client_trainer = []
    models=[]
    global_logits = []
    client_acc_list = {}
    for i in range(0, client_number):
        model_client = ResNet18()
        models.append(model_client)
    for i in range(0,client_number):
        models[i].to(device)
    for i in range(0, client_number):
        client_trainer.append(clientTrainer.ClientTrainer(i, client_loaders,public_loader,device,
                                               models[i], test_loader, global_logits))
    for i in range(0, client_number):
        client_acc_list[i] = []
    for round in range (0,num_rounds):
        print(f"start round {round}")
        for i in range(0, client_number):
            client_trainer[i].pretrain(10)
        adjacency_matrix = generate_adjacency_matrix(client_number)
        nearest_adjacency_matrix = find_n_smallest_nonzero_distances(adjacency_matrix,10)
        for i in range(0, client_number):
            logits_dict=client_trainer[i].present()
            for j in range(len(nearest_adjacency_matrix[i])):
                # 实现找出adjacency_matrix[i]中最小的n个距离
                client_trainer[nearest_adjacency_matrix[i][j]].add_local_trained_result(i, logits_dict)
        for i in range(0, client_number):
            logits = client_trainer[i].average(client_trainer[i].client_logits)
            client_trainer[i].update_large_model_logits(logits)
            client_trainer[i].client_logits = []
            print(f"sent back to client{i}")
        for i in range(0, client_number):
            logits_dict, acc_list = client_trainer[i].train(num_epochs)
            print("finish training client")
            for j in range(len(nearest_adjacency_matrix[i])):
                # 实现找出adjacency_matrix[i]中最小的n个距离
                client_trainer[nearest_adjacency_matrix[i][j]].add_local_trained_result(i, logits_dict)
                print(len(client_trainer[nearest_adjacency_matrix[i][j]].client_logits))
            if len(client_acc_list[i]) == 0:
                client_acc_list[i] = acc_list
            else:
                client_acc_list[i].extend(acc_list)
        round_idx += 1
        for i in range (0,client_number):
            logits = client_trainer[i].average(client_trainer[i].client_logits)
            client_trainer[i].update_large_model_logits(logits)
            client_trainer[i].client_logits = []
            print(f"sent back to client{i}")
    file_path = "cifar10-20client-1.json"
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(client_acc_list, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


