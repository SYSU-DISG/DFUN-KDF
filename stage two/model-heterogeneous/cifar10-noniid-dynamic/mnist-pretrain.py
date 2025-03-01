import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from resnet18 import ResNet18
from  models import *
# 定义 ResNet-18 模型
device = 'cuda'

# 准备数据集
transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
models = []
# 初始化 10 个相同的 ResNet-18 模型
for i in range(0, 10):
    model_client = CNN_3layer_fc_model_removelogsoftmax()
    models.append(model_client)
for i in range(10):
    models[i].to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion.to(device)


# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # 训练每个模型
    for model_idx, model in enumerate(models):
        model.train()
        running_loss = 0.0
        print(f"Training Model {model_idx + 1}")
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        for batch_idx, (images, labels) in enumerate(train_loader):
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
        loss_avg = running_loss / len(train_loader)
    # 保存每个模型
for model_idx, model in enumerate(models):
    save_path = f'./model/model_{model_idx + 1}_pretrain_{epoch + 1}.pt'
    torch.save(model.state_dict(), save_path)
    print(f"ResNet-18 Model {model_idx + 1} saved at {save_path}")
