import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import *
import argparse
from resnet18 import *
import os
import Loss
from torch.utils.data import random_split
#
import clientTrainer
import copy
from dataprocess import *
from adjacency import *
import json
import numpy
from uavhuman import *
dirichlet_alpha = 0.5
torch.manual_seed(42)
numpy.random.seed(42)
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多

# 超参数设置
# 准备数据集并预处理
device = 'cuda'
# 加载 EMNIST 数据集
train_dataset, test_dataset = get_uavhuman(root='uavhuman',
                                               transforms=transforms.Compose([
                                                   transforms.Resize((256, 256)),
                                                   transforms.ToTensor()
                                               ]))
publicdata,clientdatas = random_split(train_dataset,[1003,len(train_dataset)-1003])
print(len(publicdata))
print(len(clientdatas))
client_number=10
BATCH_SIZE=32
transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
client_data_size = len(clientdatas) // client_number
train_labels = []
for i in range(len(clientdatas)):
    train_labels.append(clientdatas[i][3])
train_labels = torch.tensor(train_labels)
client_datasets = get_dataset(clientdatas,client_number,dirichlet_alpha,train_labels)
# client_datasets = random_split(clientdatas, [client_data_size] * client_number)
for i in range(client_number):
    print(len(client_datasets[i]))
# 创建 DataLoader
client_loaders = [torch.utils.data.DataLoader(dataset, BATCH_SIZE,shuffle=True) for dataset in client_datasets]
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
public_loader = torch.utils.data.DataLoader(publicdata, batch_size=128, shuffle=False)
num_rounds=1
# 定义损失函数和优化方式
def main():
    round_idx = 0
    num_epochs=5
    client_trainer = []
    models=[]
    global_logits = []
    client_acc_list = {}
    for i in range(0, 10):
        model_client = SimpleResNet18()
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
            client_acc_list[i] = client_trainer[i].pretrain(100)
        round_idx += 1

    file_path = "UAVhuman-noniid-3-1.json"
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(client_acc_list, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


