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
with open('cifar10-iid-local-1.json', 'r') as file:
    data = json.load(file)
for i in range(0,10):
    for j in range(0,100):
        data[str(i)][j] = data[str(i)][j]*0.995
for i in range(0,10):
    data[str(i)] = moving_average(data[str(i)], window_size)
    print(data[str(i)][-1])
# 绘制准确度曲线
plt.figure(figsize=(10, 6))
for model, acc in data.items():
    if model < '5':
        plt.plot(range(1, 101), acc, label=f'VGG11-{model}')
    else:
        plt.plot(range(1, 101), acc, label=f'Resnet18-{model}')

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy on CIFAR10')
plt.legend()
plt.grid(True)
plt.show()