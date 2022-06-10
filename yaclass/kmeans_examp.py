'''
Kmeans里最主要的是找中心点

1.明确输入、输出分别是什么，怎么用数学来表达
2.初始化：类别数K=？、类中心
3.所有的点points0都和当前选中的中心点分别计算
一次距离，然后判断离哪个中心点距离最近，就归到哪
一类
k1：{pi，pj，……pm}  Ω1  means1=（（xi+xj+））
k2：{pa，pb，……}     Ω2
k3：{px，py……}         Ω3
4.更新中心点
通过求平均值更新中心点
Ω1  means1=（（xi+xj+…xm）/m,(yi+yj+…ym)/m））
Ω2
Ω3
5.当前的中心点和前一次中心点非常接近
这时候就是我们要得到的中心点，停止迭代了
'''
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS
from collections import defaultdict
points0 = np.random.normal(size=(100,2))

def random_centers(k,points):
    for i in range(k):
        yield random.choice(points[:,0]) ,random.choice(points[:,1])
        #print(random.choice(points[:,0]),random.choice(points[:,1]))

def distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def mean(points):
    all_x,all_y = [x for x,y in points],[y for x,y in points]
    #x_m=np.mean(all_x)
    #y_m=np.mean(all_y)
    #return x_m,y_m
    return np.mean(all_x),np.mean(all_y)

def kmeans(K,points,centers=None):
    colors = list(BASE_COLORS.values())
    if not centers:
        centers= list(random_centers(K,points))
    for i,c in enumerate(centers):
        #print(i,c)
        plt.scatter([c[0]],[c[1]],s=90,marker='*',c=colors[i])
    plt.scatter(*zip(*points),c='black')
    plt.show()
    centers_neighbor = defaultdict(set)
    #计算距离、分组
    for p in points:
        #求最小值，离每个center的距离最小的点
        closet_c = min(centers, key=lambda centers:distance(p, centers))
        #把每一组用字典表达出来，key是closet_c
        centers_neighbor[closet_c].add(tuple(p))
    #把每一组点用不同颜色画出来
    for i,c in enumerate(centers):
        _points = centers_neighbor[c]
        all_x,all_y = [x for x,y in _points],[y for x,y in _points]
        plt.scatter(all_x,all_y,c=colors[i])

    plt.show()
    #更新中心点
    new_centers=[]
    for c in centers_neighbor:
        new_c = mean(centers_neighbor[c])
        new_centers.append(new_c)

    #判断前后两次中心点的距离，如果小于阈值就停止，否则继续迭代
    distances_old_and_new = [distance(c_old,c_new) for c_old,c_new in zip(centers,new_centers)]
    print(distances_old_and_new)
    threshold=0.1
    if all(c<threshold for c in distances_old_and_new):
        return centers_neighbor
    else:
        kmeans(K,points,new_centers)
kmeans(3,points0)