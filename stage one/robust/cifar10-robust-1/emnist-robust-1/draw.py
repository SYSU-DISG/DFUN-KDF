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
with open('emnist-robust-1.json', 'r') as file:
    data1 = json.load(file)
with open('emnist-robust-2.json', 'r') as file:
    data2 = json.load(file)
with open('emnist-robust-3.json', 'r') as file:
    data3 = json.load(file)
with open('emnist-robust-4.json', 'r') as file:
    data4 = json.load(file)
with open('emnist-robust-5.json', 'r') as file:
    data5 = json.load(file)
print(len(data1))
# for i in range(0,10):
#     for j in range(0,30):
#         data1[str(i)][j] = data1[str(i)][j] * 0.99
#         data1[str(i)][99-j] = data1[str(i)][99-j] * 1.01
# for i in range(0,10):
#     for j in range(0,20):
#         data2[str(i)][j] = data1[str(i)][j] * 0.99
#         data2[str(i)][99-j] = data2[str(i)][99-j] * 1.005

for i in range(len(data1)):
    data1[str(i)] = moving_average(data1[str(i)], window_size)
for i in range(len(data2)):
    data2[str(i)] = moving_average(data2[str(i)], window_size)
for i in range(len(data3)):
    data3[str(i)] = moving_average(data3[str(i)], window_size)
for i in range(len(data4)):
    data4[str(i)] = moving_average(data4[str(i)], window_size)
for i in range(len(data5)):
    data5[str(i)] = moving_average(data5[str(i)], window_size)

data1 = [sum(values) / len(values) for values in zip(*data1.values())]
data2 = [sum(values) / len(values) for values in zip(*data2.values())]
data3 = [sum(values) / len(values) for values in zip(*data3.values())]
data4 = [sum(values) / len(values) for values in zip(*data4.values())]
data5 = [sum(values) / len(values) for values in zip(*data5.values())]

data1=moving_average(data1,5)
data2=moving_average(data2,5)
data3=moving_average(data3,5)
data4=moving_average(data4,5)
data5=moving_average(data5,5)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), data1, label=f'crashing1')
plt.plot(range(1, 101), data2, label=f'crashing2')
plt.plot(range(1, 101), data3, label=f'crashing3')
plt.plot(range(1, 101), data4, label=f'crashing4')
plt.plot(range(1, 101), data5, label=f'crashing5')


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# plt.ylim(50, 90)
plt.show()