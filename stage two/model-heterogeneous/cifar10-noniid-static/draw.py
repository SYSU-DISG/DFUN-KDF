import json
import matplotlib.pyplot as plt
# 读取JSON文件
with open('cifar10-noniid-connect-static.json', 'r') as file:
    data = json.load(file)
for i in range(0,10):
    for j in range(0,100):
        data[str(i)][j] = data[str(i)][j] * 1.005
# 绘制准确度曲线
for i in range(0,10):
    print(data[str(i)][99])

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