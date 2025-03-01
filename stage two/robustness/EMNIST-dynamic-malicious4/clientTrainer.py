import statistics

import torch
from torch import nn,optim
import Loss
import torch.nn.functional as F
import copy
import numpy as np
class ClientTrainer(object):
    def __init__(self, client_index, local_training_data,public_data, device,
                 client_model,testloader,global_logits):
        self.client_index = client_index
        self.local_training_data = local_training_data[client_index]
        self.public_data = public_data
        self.client_logits = []
        self.flag_client_model_uploaded_dict = dict()
        # self.local_sample_number = local_sample_number
        self.device = device
        self.client_model = client_model
        self.client_model.to(self.device)
        self.testdata=testloader
        self.criterion_CE = nn.CrossEntropyLoss()
        self.model_params = self.master_params = self.client_model.parameters()
        self.global_logits = global_logits
        self.optim_params = self.master_params
        self.optimizer = torch.optim.SGD(self.optim_params, lr=0.01, momentum=0.9,
                                    weight_decay=5e-4)
        self.criterion_KL = Loss.KL_Loss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_NL = nn.L1Loss()

    def compare_dicts(self,dict1, dict2):
        # 比较字典中的每个键值对
        for key in dict1:
            # 如果键在两个字典中都存在
            if key in dict2:
                value1 = dict1[key]
                value2 = dict2[key]

                # 如果值是 NumPy 数组，使用 np.array_equal() 比较
                if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                    if not np.array_equal(value1, value2):
                        return False
                # 如果值是字典，递归比较
                elif isinstance(value1, dict) and isinstance(value2, dict):
                    if not self.compare_dicts(value1, value2):
                        return False
                # 其他情况直接使用 == 进行比较
                elif value1 != value2:
                    return False
            # 如果键只在一个字典中存在
            else:
                return False

        # 检查第二个字典是否有多余的键
        for key in dict2:
            if key not in dict1:
                return False

        # 如果所有键值对都相等，则字典相同
        return True

    # 定义损失函数和优化方式
    def update_large_model_logits(self, logits):
        self.global_logits = logits

    def add_local_trained_result(self, index, logits_dict):
        print("add_model. index = %d" % index)
        self.client_logits.append(logits_dict)
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self, client_num):
        for idx in range(client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def flatten_dict_values(self,d, flattened_values):
        for value in d.values():
            if isinstance(value, dict):
                self.flatten_dict_values(value, flattened_values)
            else:
                flattened_values.append(value)
        return flattened_values
    def pick(self,client_logits):
        flattened_value = [[] for _ in range(10)]
        for i in range(len(client_logits)):
            flattened_value[i] = self.flatten_dict_values(client_logits[i], flattened_value[i])
        for i in range(len(flattened_value)):
            print(len(flattened_value[i]))
        p = [0 for _ in range(10) ]
        for i in range(len(self.public_data)):
            l = [0 for _ in range(10)]
            for j in range(len(client_logits)):
                l[i] = flattened_value[j][i][0]
                print(l[i])
            median_value = statistics.median(l)
            median_index = l.index(median_value)
            p[median_index] += 1
        for i in range(len(p)):
            p[i] = p[i]/len(self.public_data)
        return p
    def weighted_average(self,client_logits,p,result):
        for key in client_logits[0]:
            if isinstance(client_logits[0][key], dict):
                result[key] = self.weighted_average([d[key] for d in client_logits],p)
            else:
                for i in range(len(client_logits)):
                    result[key] += client_logits[i][key]*p[i]
        return result
    def add_nested_dicts(self, dict1, dict2):
        result = {}
        for key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # 递归调用，将相应键的值相加
                result[key] = self.add_nested_dicts(dict1[key], dict2[key])
            else:
                # 如果值是可加的（例如，数字），则相加
                result[key] = dict1[key] + dict2[key]
        return result

    def aver(self, dict1, num):
        result = {}
        for key in dict1:
            if isinstance(dict1[key], dict):
                result[key] = self.aver(dict1[key])
            else:
                result[key] = dict1[key] / num
        return result

    def average(self, client_logits):
        result = {}
        for i in range(len(client_logits)):
            if len(result) != 0:
                result = self.add_nested_dicts(result, client_logits[i])
            else:
                result = copy.deepcopy(client_logits[i])
        result = self.aver(result, len(client_logits))
        return result
    def Krum(self,client_logits):
        result = copy.deepcopy(client_logits[0])
        if len(client_logits) == 1:
            return client_logits[0]
        else:
            for key in client_logits[0]:
                distance = [0 for _ in range(len(client_logits))]
                for i in range(len(client_logits)):
                    for j in range(len(client_logits)):
                        distance[i] += np.linalg.norm(client_logits[i][key]-client_logits[j][key])
                sorted_indices = sorted(range(len(distance)), key=lambda i: distance[i])[:2]
                result[key] = (client_logits[sorted_indices[0]][key]+client_logits[sorted_indices[1]][key])/2
            return result
    def pretrain(self,epochs):
        for epoch in range(epochs):
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
                loss = self.criterion_CE(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * float(correct) / total))
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
                print('client {} - Epoch {} trainning acc: {:.3f}%'.format(self.client_index, epoch, acc))
        #将模型参数保存到本地
        client_model_path = './client_model/client_model_{}.pth'.format(self.client_index)
        torch.save(self.client_model.state_dict(), client_model_path)
    def present(self):
        logits_dict = dict()
        self.client_model.eval()
        for batch_idx, (images, labels) in enumerate(self.public_data):
            images, labels = images.to(self.device), labels.to(self.device)
            log_probs = self.client_model(images)
            log_probs = log_probs.cpu().detach().numpy()
            logits_dict[batch_idx] = log_probs
        return logits_dict
    def train(self,num_epochs):
        logits_dict = dict()
        acc_list = []
        print(len(self.global_logits))
        for epoch in range(num_epochs):
            print('\nEpoch: %d' % (epoch + 1))
            self.client_model.train()
            total = 0.0
            correct = 0.0
            sum_loss = 0.0
            print(id(self.client_model))
            for batch_idx, data in enumerate(self.public_data, 0):

                # 准备数据
                length = len(self.public_data)
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward + backward
                outputs = self.client_model(inputs)
                large_model_logits = torch.from_numpy(self.global_logits[batch_idx]).to(
                    self.device)
                loss_kd = self.criterion_KL.forward(outputs, large_model_logits)
                loss_true = self.criterion_CE(outputs, labels)
                loss_mse = self.criterion_mse(outputs,large_model_logits)
                loss_l1 = self.criterion_NL(outputs,large_model_logits)
                loss = loss_kd

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
                print('client {} - Epoch {} trainning acc: {:.3f}%'.format(self.client_index, epoch, acc))
                acc_list.append(acc.item())
        self.client_model.eval()
        for batch_idx, (images, labels) in enumerate(self.public_data):
            images, labels = images.to(self.device), labels.to(self.device)
            log_probs = self.client_model(images)
            log_probs = log_probs.cpu().detach().numpy()
            logits_dict[batch_idx] = log_probs
        return logits_dict,acc_list

