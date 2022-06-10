import numpy as np
import matplotlib.pyplot as plt

def p(a):
    print(a)

def ya(x, l):
    # 这一节对gamma先不做探讨，先定为1
    gamma = 1.0
    # 这里x-l是一个数，不是向量，所以不需要取模
    return np.exp(-gamma * (x - l) ** 2)



def ya1():
    # 构建样本数据，x值从-4到5，每个数间隔为1
    x = np.arange(-4, 5, 1)
    # y构建为0，1向量，且是线性不可分的
    y = np.array((x >= -2) & (x <= 2), dtype='int')
    # 绘制样本数据
    plt.scatter(x[y == 1], [0] * len(x[y == 1]))
    plt.scatter(x[y == 0], [0] * len(x[y == 0]))
    plt.show()
def ya2():
    # 将每一个x值通过高斯核函数和l1，l2地标转换为2个值，构建成新的样本数据
    x = np.arange(-4, 5, 1)
    y = np.array((x >= -2) & (x <= 2), dtype='int')
    l1, l2 = -1, 1
    X_new = np.empty((len(x), 2))
    #p(X_new)
    for i, data in enumerate(x):
        X_new[i, 0] = ya(data, l1)
        X_new[i, 1] = ya(data, l2)
    # 绘制新的样本点
    plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1])
    plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1])
    plt.show()
if __name__ == '__main__':
    ya2()