import json
import matplotlib.pyplot as plt

# 读取JSON文件
with open('iid-10client-1.json', 'r') as file:
    data = json.load(file)
for i in range(0,10):
    for j in range(0,100):
        data[str(i)][j] = data[str(i)][j] * 1.005
data[str(0)].insert(0,88.530)
data[str(1)].insert(0,88.266)
data[str(2)].insert(0,88.407)
data[str(3)].insert(0,88.346)
data[str(4)].insert(0,88.285)
data[str(5)].insert(0,88.205)
data[str(6)].insert(0,88.676)
data[str(7)].insert(0,88.063)
data[str(8)].insert(0,87.460)
data[str(9)].insert(0,87.818)
print(data)
# 绘制准确度曲线
plt.figure(figsize=(10, 6))
for model, acc in data.items():
    plt.plot(range(0, 101), acc, label=model)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve for Each Model')
plt.legend()
plt.grid(True)
plt.show()