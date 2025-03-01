import torch
import torch.nn as nn
import torchvision.models as models

class SimpleResNet18(nn.Module):
    def __init__(self, log_prob=6):
        super(SimpleResNet18, self).__init__()
        # 加载预训练的 ResNet-18 模型，去掉最后的全连接层
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        # 添加自定义的全连接层
        self.fc1 = nn.Linear(512, log_prob)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.fc1(x)
        return output