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
print(emnist_test)
client_number=10
BATCH_SIZE=128
transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
# 选择标签为a到f的数据

labels_a_to_f = [i for i in range(1, 7)]  # a=97, b=98, ..., f=102
train_indices = [i for i in range(len(emnist_train)) if emnist_train.targets[i] in labels_a_to_f]
#打印train_indices前十个标签

test_indices = [i for i in range(len(emnist_test)) if emnist_test.targets[i] in labels_a_to_f]
# 创建子集
emnist_train.targets+=9
emnist_test.targets+=9
client_data_size = len(emnist_train) // client_number
client_datasets = random_split(emnist_train, [client_data_size] * client_number)
# 创建 DataLoader
client_loaders = [torch.utils.data.DataLoader(dataset, BATCH_SIZE) for dataset in client_datasets]
print(len(client_loaders))
test_loader = torch.utils.data.DataLoader(emnist_test, batch_size=100, shuffle=True)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
# 打印数据集大小
mnist_test_loader = DataLoader(datasets.MNIST(root='./data', train=True, transform=transform, download=True), batch_size=128, shuffle=True)
models = []
for i in range(0, 10):
    model_client = CNN_3layer_fc_model_removelogsoftmax()
    models.append(model_client)
for i in range(10):
    models[i].load_state_dict(torch.load('./model/model_{}_pretrain_5.pt'.format(i+1)))
    models[i].to(device)
def pretest(models):
    for model_idx, model in enumerate(models):
        with torch.no_grad():
            correct = 0
            total = 0
            for data in mnist_test_loader:
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                acc = 100. * correct / total
            print('client {}  mnist test acc: {:.3f}%'.format(model_idx, acc))

def train():
    num_epochs = 2
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
                for data in mnist_test_loader:
                    model.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    acc = 100. * correct / total
                print('client {} - Epoch {} mnist test acc: {:.3f}%'.format(model_idx, epoch, acc))
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
    pretest(models)
    acc_list = train()
    print("Training Finished")
    for model_idx, model in enumerate(models):
        save_path = f'./model/test1_emnist_model_{model_idx + 1}_pretrain.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Model {model_idx + 1} saved at {save_path}")
    # 保存acc_list
    with open('./model/acc_list_pretrain.json', 'w') as f:
        json.dump(acc_list, f)