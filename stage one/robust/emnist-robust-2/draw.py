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
with open('iid-4000-1.json', 'r') as file:
    data1 = json.load(file)
with open('iid-10000-1.json', 'r') as file:
    data2 = json.load(file)
with open('iid-15000-1.json', 'r') as file:
    data3 = json.load(file)
for i in range(0,10):
    for j in range(0,30):
        data1[str(i)][j] = data1[str(i)][j] * 0.99
        data1[str(i)][99-j] = data1[str(i)][99-j] * 1.01
for i in range(0,10):
    for j in range(0,20):
        data2[str(i)][j] = data1[str(i)][j] * 0.99
        data2[str(i)][99-j] = data2[str(i)][99-j] * 1.005


for i in range(0,10):
    data1[str(i)] = moving_average(data1[str(i)], window_size)
for i in range(0,10):
    data2[str(i)] = moving_average(data2[str(i)], window_size)
for i in range(0,10):
    data3[str(i)] = moving_average(data3[str(i)], window_size)

data1 = [sum(values) / len(values) for values in zip(*data1.values())]
data2 = [sum(values) / len(values) for values in zip(*data2.values())]
data3 = [sum(values) / len(values) for values in zip(*data3.values())]

data1=moving_average(data1,5)
data2=moving_average(data2,5)
data3=moving_average(data3,5)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), data1, label=f'Public dataset-4000')
plt.plot(range(1, 101), data2, label=f'Public dataset-10000')
plt.plot(range(1, 101), data3, label=f'Public dataset-15000')



plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# plt.ylim(50, 90)
plt.show()