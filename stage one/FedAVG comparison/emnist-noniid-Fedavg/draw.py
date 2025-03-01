import json
import matplotlib.pyplot as plt

# 读取JSON文件
with open('emnist-noniid-Fedavg-1-2.json', 'r') as file:
    data = json.load(file)

print(data)
# 绘制准确度曲线
plt.figure(figsize=(10, 6))
for model, acc in data.items():
    plt.plot(range(1, 101), acc, label=model)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve for Each Model')
plt.legend()
plt.grid(True)
plt.show()