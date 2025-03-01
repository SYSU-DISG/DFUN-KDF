import json
import matplotlib.pyplot as plt

# 读取JSON文件
with open('emnist-noniid-Fedavg-dynamic.json', 'r') as file:
    data = json.load(file)

for i in range(0,10):
    for j in range(0,100):
        data[str(i)][j] = data[str(i)][j]
# 绘制准确度曲线
plt.figure(figsize=(10, 6))
for model, acc in data.items():
    plt.plot(range(1, 101), acc, label=model)
for i in range(0,10):
    print(data[str(i)][99])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve for Each Model')
plt.legend()
plt.grid(True)
plt.show()