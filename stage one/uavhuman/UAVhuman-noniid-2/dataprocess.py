import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torch.distributions.dirichlet import Dirichlet
import torch


def iid_divide(l, g):
    """
    将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
    每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
    返回由不同的groups组成的列表
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist




def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client):
    data_idcs = list(range(len(dataset)))
    label2index = {k: [] for k in range(n_classes)}
    for idx in data_idcs:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]
    n_shards = n_clients * n_classes_per_client
    # 一共分成n_shards个独立同分布的shards
    shards = iid_divide(sorted_idcs, n_shards)
    np.random.shuffle(shards)
    # 然后再将n_shards拆分为n_client份
    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            # 这里shard是一个shard的数据索引(一个列表)
            # += shard 实质上是在列表里并入列表
            clients_idcs[client_id] += shard
    return clients_idcs


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs




def get_dataset(trainset,client_num,alpha,train_labels):
    '''
    从训练集中划分出client_num个client，每个client的数据量服从参数为alpha的Dirichlet分布
    '''
    client_idcs = dirichlet_split_noniid(train_labels,alpha,client_num)
    client_datasets = [torch.utils.data.Subset(trainset.dataset,idcs) for idcs in client_idcs]
    return client_datasets

# import numpy
# from uavhuman import get_uavhuman
# from torch.utils.data import random_split
# from torchvision import transforms
# dirichlet_alpha = 0.5
# torch.manual_seed(42)
# numpy.random.seed(42)
# # 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
# train_labels = []
#
# # 超参数设置
# # 准备数据集并预处理
# device = 'cuda'
# # 加载 EMNIST 数据集
# train_dataset, test_dataset = get_uavhuman(root='uavhuman',
#                                                transforms=transforms.Compose([
#                                                    transforms.Resize((256, 256)),
#                                                    transforms.ToTensor()
#                                                ]))
# publicdata,clientdatas = random_split(train_dataset,[1003,len(train_dataset)-1003])
# print(len(publicdata),len(clientdatas) )
# print(type(clientdatas[0][0]))
# print(type(clientdatas[0][1]))
# print(type(clientdatas[0]))
# datas = torch.tensor([])
# for i in range(len(clientdatas)):
#     train_labels.append(clientdatas[i][3])
# train_labels = torch.tensor(train_labels)
# client_number=10
# BATCH_SIZE=128
# transform = transforms.Compose([transforms.ToTensor(),
#                                           transforms.Normalize((0.1307,), (0.3081,))])
# client_data_size = len(clientdatas) // client_number
# client_datasets = get_dataset(clientdatas,client_number,dirichlet_alpha,train_labels)
# for i in range(client_number):
#     print(len(client_datasets[i]))
#     print(client_datasets[i][0])