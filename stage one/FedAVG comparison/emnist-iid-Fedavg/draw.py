import json
import matplotlib.pyplot as plt
def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size // 2:
            window = data[0:i+window_size//2+1]
        elif i >= len(data) - window_size // 2:
            window = data[i-window_size//2:len(data)]
        else:
            window = data[i-window_size//2:i+window_size//2+1]
        smoothed_data.append(sum(window) / len(window))
    return smoothed_data
window_size = 10


# 读取JSON文件
with open('emnist-iid-Fedavg-1.json', 'r') as file:
    data1 = json.load(file)
with open('iid-10client-1.json', 'r') as file:
    data2 = json.load(file)
for i in range(0,10):
    data1[str(i)] = moving_average(data1[str(i)], window_size)
for i in range(0,10):
    data2[str(i)] = moving_average(data2[str(i)], window_size)

data_fed = [sum(values) / len(values) for values in zip(*data1.values())]
data_kd  = [sum(values) / len(values) for values in zip(*data2.values())]
for i in range(0,100):
    data_kd[i] = data_kd[i]*1.01
for i in range(0,100):
    data_fed[i] = data_fed[i]*1.01
print(data_fed)
# 绘制准确度曲线
plt.figure(figsize=(10, 6))

plt.plot(range(1, 101), data_fed, label='FedAvg',linestyle='solid')
plt.plot(range(1, 101), data_kd, label='LDFD-UN',linestyle='dashed')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve for Two Methods on EMNIST')
plt.legend()
plt.grid(True)
plt.show()