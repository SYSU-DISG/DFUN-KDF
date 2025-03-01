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
client_number=10
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
test_loader = torch.utils.data.DataLoader(emnist_test, batch_size=100, shuffle=True)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
public_loader = torch.utils.data.DataLoader(publicdata, batch_size=128, shuffle=False)
models=[]
for i in range(0, 5):
    model_client = CNN_3layer_fc_model_removelogsoftmax()
    models.append(model_client)
for i in range(0, 5):
    model_client = CNN_2layer_fc_model_removelogsoftmax()
    models.append(model_client)
for i in range(10):
    models[i].to(device)
num_rounds= 20
# 定义损失函数和优化方式
def train():
    num_epochs = 100
    client_acc_list = {}
    for i in range(10):
        client_acc_list[i] = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for model_idx, model in enumerate(models):
            criterion = nn.CrossEntropyLoss()
            criterion.to(device)
            model.train()
            running_loss = 0.0
            print(f"Training Model {model_idx + 1}")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                            weight_decay=5e-4)
            for batch_idx, (images, labels) in enumerate(client_loaders[model_idx]):
                correct = 0
                total = 0
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                acc = correct / total
                acc = acc * 100
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} Loss: {} acc:{}%'.format(
                        epoch + 1, loss.item(), acc ))
                running_loss += loss.item()
            loss_avg = running_loss / len(client_loaders[model_idx])
        #进行测试
            with torch.no_grad():
                correct = 0
                total = 0
                for data in test_loader:
                    model.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    acc1 = 100. * correct / total
                print('client {} - Epoch {} trainning acc: {:.3f}%'.format(model_idx, epoch, acc1))
            client_acc_list[model_idx].append(acc1.item())
    return client_acc_list
if __name__ == '__main__':
    client_acc_list = train()
    file_path = "emnist-iid-local-1.json"
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(client_acc_list, json_file, ensure_ascii=False, indent=2)