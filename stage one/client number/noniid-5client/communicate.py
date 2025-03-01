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

# 超参数设置
# 准备数据集并预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
device = 'cuda'
# 加载 EMNIST 数据集
emnist_train = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
emnist_test = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
emnist_train.targets = emnist_train.targets - 1
emnist_test.targets = emnist_test.targets - 1
publicdata,clientdatas = random_split(emnist_train,[4000,len(emnist_train)-4000])
print(len(publicdata))
print(len(clientdatas))
client_number=5
BATCH_SIZE=128
transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
client_data_size = len(clientdatas) // client_number
# client_datasets = random_split(clientdatas, [client_data_size] * client_number)
client_datasets = get_dataset(clientdatas, client_number, dirichlet_alpha)
for i in range(0, client_number):
    print(len(client_datasets[i]))
# 创建 DataLoader
client_loaders = [torch.utils.data.DataLoader(dataset, BATCH_SIZE,shuffle=True) for dataset in client_datasets]
test_loader = torch.utils.data.DataLoader(emnist_test, batch_size=100, shuffle=True)
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
    for i in range(0, 10):
        model_client = CNN_3layer_fc_model_removelogsoftmax()
        models.append(model_client)
    for i in range(10):
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
        nearest_adjacency_matrix = find_n_smallest_nonzero_distances(adjacency_matrix,3)
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
    file_path = "noniid-5client-1.json"
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(client_acc_list, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


