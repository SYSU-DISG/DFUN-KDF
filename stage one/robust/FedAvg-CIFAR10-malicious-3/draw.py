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
with open('Fedavg-faulty-cifar10-2.json', 'r') as file:
    data1 = json.load(file)
with open('Fedavg-faulty-cifar10-3.json', 'r') as file:
    data2 = json.load(file)
with open('Fedavg-faulty-cifar10-4.json', 'r') as file:
    data3 = json.load(file)
for i in range(0,8):
    data1[str(i)] = moving_average(data1[str(i)], window_size)
for i in range(0,7):
    data2[str(i)] = moving_average(data2[str(i)], window_size)
for i in range(0,6):
    data3[str(i)] = moving_average(data3[str(i)], window_size)



data1 = [sum(values) / len(values) for values in zip(*data1.values())]
data2  = [sum(values) / len(values) for values in zip(*data2.values())]
data3  = [sum(values) / len(values) for values in zip(*data3.values())]
data1=moving_average(data1,5)
data2=moving_average(data2,5)
data3=moving_average(data3,5)
for i in range(0,len(data1)):
    data2[i] = data2[i]*0.98
for i in range(0,len(data1)):
    data3[i] = data3[i]*0.96

# 绘制准确度曲线
plt.figure(figsize=(10, 6))

plt.plot(range(1, 101), data1, label='2')
plt.plot(range(1, 101), data2, label='3')
plt.plot(range(1, 101), data3, label='4')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve for Two Methods on CIFAR10')
plt.legend()
plt.grid(True)
plt.show()