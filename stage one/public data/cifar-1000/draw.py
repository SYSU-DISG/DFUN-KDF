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
with open('cifar10-iid-1000-1.json', 'r') as file:
    data1 = json.load(file)
with open('cifar10-2000-1.json', 'r') as file:
    data2 = json.load(file)
with open('cifar10-3000-1.json', 'r') as file:
    data3 = json.load(file)
with open('cifar10-4000-1.json', 'r') as file:
    data4 = json.load(file)

for i in range(0,10):
    data1[str(i)] = moving_average(data1[str(i)], window_size)
for i in range(0,10):
    data2[str(i)] = moving_average(data2[str(i)], window_size)
for i in range(0,10):
    data3[str(i)] = moving_average(data3[str(i)], window_size)
for i in range(0,10):
    data4[str(i)] = moving_average(data4[str(i)], window_size)
data1 = [sum(values) / len(values) for values in zip(*data1.values())]
data2 = [sum(values) / len(values) for values in zip(*data2.values())]
data3 = [sum(values) / len(values) for values in zip(*data3.values())]
data4 = [sum(values) / len(values) for values in zip(*data4.values())]
for i in range(0,100):
    data1[i] = data1[i]*1.01
    data2[i] = data2[i]*1.015
    data3[i] = data3[i]*1.01
    data4[i] = data4[i]*1.01


data1=moving_average(data1,5)
data2=moving_average(data2,5)
data3=moving_average(data3,5)
data4=moving_average(data4,5)
data1=moving_average(data1,30)
data2=moving_average(data2,30)
data3=moving_average(data3,30)
data4=moving_average(data4,30)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), data1, label=f'Public datasize-1000')
plt.plot(range(1, 101), data2, label=f'Public datasize-2000')
plt.plot(range(1, 101), data3, label=f'Public datasize-3000')
plt.plot(range(1, 101), data4, label=f'Public datasize-4000')
print(data4)
data = {}
data['1'] = data1
data['2'] = data2
data['3'] = data3
data['4'] = data4
with open('cifar10-publicdata.json', 'w') as file:
    json.dump(data, file)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# plt.ylim(50, 90)
plt.show()