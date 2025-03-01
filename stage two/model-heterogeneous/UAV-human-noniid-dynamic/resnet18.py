import torch
import torch.nn as nn
import torchvision.models as models

class SimpleResNet18(nn.Module):
    def __init__(self, log_prob=6):
        super(SimpleResNet18, self).__init__()
        # 加载预训练的 ResNet-18 模型，去掉最后的全连接层
        resnet18 = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        # 添加自定义的全连接层
        self.fc1 = nn.Linear(512, log_prob)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.fc1(x)
        return output
class AlexNet6(nn.Module):
    def __init__(self):
        super(AlexNet6, self).__init__()
        self.alexnet = models.alexnet(pretrained=False)
        self.num_classes = 6
        # 替换最后一层的全连接层
        self.alexnet.classifier[6] = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        return self.alexnet(x)