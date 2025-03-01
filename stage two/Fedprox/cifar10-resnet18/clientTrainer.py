import torch
from torch import nn,optim
import Loss
import torch.nn.functional as F
import copy
import numpy as np
class ClientTrainer(object):
    def __init__(self, client_index, local_training_data, device,
                 client_model,testloader):
        self.client_index = client_index
        self.local_training_data = local_training_data[client_index]
        self.client_models = []
        self.flag_client_model_uploaded_dict = dict()
        # self.local_sample_number = local_sample_number
        self.device = device
        self.client_model = client_model
        self.client_model.to(self.device)
        self.testdata=testloader
        self.criterion_CE = nn.CrossEntropyLoss()
        self.model_params = self.master_params = self.client_model.parameters()
        self.optim_params = self.master_params
        self.optimizer = torch.optim.SGD(self.optim_params, lr=0.01, momentum=0.9,
                                    weight_decay=5e-4)
        self.criterion_KL = Loss.KL_Loss(temperature=1)
        self.mu = 0.02
        self.global_model_params = self.client_model.parameters()
    def update_large_model_parameters(self,global_model):
        self.client_model.load_state_dict(global_model)

    def add_local_trained_result(self, index, models):
        print("add_model. index = %d" % index)
        self.client_models.append(models)
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self, client_num):
        for idx in range(client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def average_parameters(self,list_of_state_dicts):
        # 确定模型数量
        num_models = float(len(list_of_state_dicts))

        # 使用第一个模型的 state_dict 键初始化平均 state_dict
        averaged_state_dict = {key: 0.0 for key in list_of_state_dicts[0].keys()}

        # 将所有模型的值累加到平均 state_dict 中
        for state_dict in list_of_state_dicts:
            for key in state_dict.keys():
                averaged_state_dict[key] += state_dict[key]

        # 计算平均值
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] /= num_models

        return averaged_state_dict

    def train(self,num_epochs):
        logits_dict = dict()
        acc_list = []
        for epoch in range(num_epochs):
            print('\nEpoch: %d' % (epoch + 1))
            self.client_model.train()
            total = 0.0
            correct = 0.0
            sum_loss = 0.0
            print(id(self.client_model))
            for batch_idx, data in enumerate(self.local_training_data, 0):

                # 准备数据
                length = len(self.local_training_data)
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward + backward
                outputs = self.client_model(inputs)

                loss = self.criterion_CE(outputs,labels)

                prox_reg = 0.0
                for param, global_param in zip(self.client_model.parameters(), self.global_model_params):
                    prox_reg += ((param - global_param) ** 2).sum()
                loss += (self.mu / 2) * prox_reg

                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (
                      epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))
            # 每训练完一个epoch测试一下准确率
            with torch.no_grad():
                correct = 0
                total = 0
                for data in self.testdata:
                    self.client_model.eval()
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.client_model(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    acc = 100. * correct / total
                print('client {} - Epoch {} Test acc: {:.3f}%'.format(self.client_index, epoch, acc))
                acc_list.append(acc.item())
        return self.client_model.state_dict(),acc_list

