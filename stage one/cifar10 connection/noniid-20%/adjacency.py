import numpy as np


def generate_adjacency_matrix(num_nodes, max_weight=100, min_weight=1):
    """
    生成一个完全且连通的随机邻接矩阵

    Parameters:
        num_nodes (int): 节点数量
        max_weight (int): 边的最大权重，默认为100
        min_weight (int): 边的最小权重，默认为1

    Returns:
        numpy.ndarray: 随机生成的邻接矩阵
    """
    # 初始化邻接矩阵，所有元素设为0
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 随机生成完全图
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 随机生成边的权重
            weight = np.random.randint(min_weight, max_weight + 1)
            # 在无向图中，矩阵是对称的
            adjacency_matrix[i, j] = weight
            adjacency_matrix[j, i] = weight
    return adjacency_matrix


def find_n_smallest_nonzero_distances(adjacency_matrix, n):
    min_indices = np.argsort(adjacency_matrix, axis=1)
    near_adjacency = []
    for i in range(len(min_indices)):
        min_indice = []
        for j in range(0,n):
            min_indice.append(min_indices[i][j])
        near_adjacency.append(min_indice)
    near_adjacency = np.array(near_adjacency)

    return near_adjacency


for i in range(0, client_number):
    adjacency_matrix = generate_adjacency_matrix(client_number)
    nearest_adjacency_matrix = find_n_smallest_nonzero_distances(adjacency_matrix, m)
for i in range(0, client_number):
    for j in range(len(nearest_adjacency_matrix[i])):
        # 实现找出adjacency_matrix[i]中最小的n个距离
        client_trainer[nearest_adjacency_matrix[i][j]].add_local_trained_result(i, logits_dict)
for i in range(0, client_number):
    logits = client_trainer[i].average(client_trainer[i].client_logits)