import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')
width = 0.25
plt.ylim(0.55, 0.95)
x_index = ['CIFAR10', 'EMNIST', 'UAV-Human']
x_indexes = np.arange(len(x_index))
C = [0.814,0.923,0.744]
B = [0.815,0.924,0.745]
A = [0.751,0.881,0.711]
plt.bar(x_indexes - width, A,color= '#444444', width=width,label='Local Training')
plt.bar(x_indexes, B,color= '#008fd5', width=width, label='DFL-UN-Dynamic')
plt.bar(x_indexes + width, C,color= '#e5ae38', width=width, label='DFKD-UN-Dynamic')

plt.legend()
plt.xticks(ticks=x_indexes, labels=x_index)

plt.ylabel('Average Accuracy')

plt.tight_layout()
plt.show()
