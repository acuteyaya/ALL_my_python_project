'''
File Name          : homework.c
Author             : CUTEYAYA
Version            : V1.0.0
Created on         : 12/12/2021
'''

import numpy as np
def homework1(v,m,x1,x2,x3):
    print("作业1 方法一:")
    sum=0
    m1=int(v/x1)
    m2=int(v/x2)
    if m1>m1:
        m1=m
    if m2>m:
        m2=m
    for i in range(0,m1+1):
        for j in range(0,m2+1):
            if (m-i-j)>=0 and i*x1+j*x2+(m-i-j)*x3==v :
                sum=sum+1
                print("方案 ",end='')
                print(sum,end='')
                print(":")
                print("公鸡:",end='')
                print(i)
                print("母鸡:",end='')
                print(j)
                print("小鸡:",end='')
                print(m-i-j)
                print()
    print("总方案：", end='')
    print(sum)

    print("作业1 方法二（numpy优化）:")
    sum1 = 0
    for i in range(0, int(v / x1) + 1):
        a = np.array([[x2, x3], [1, 1]])
        b = np.array([v-x1*i, m-i])
        c = np.linalg.inv(a).dot(b)
        if c[0]>=0 and c[1]>=0:
            sum1 = sum1 + 1
            print("方案 ", end='')
            print(sum1, end='')
            print(":")
            print("公鸡:", end='')
            print(i)
            print("母鸡:", end='')
            print(int(c[0]))
            print("小鸡:", end='')
            print(int(c[1]))
            print()
    print("总方案：", end='')
    print(sum1)
    print()

def homework2():
    print("作业2")
    for i in range(0,10):
        for j in range(0, 10):
            print("*", end='')
        print()

if __name__=='__main__':
    
   # homework1(100,100,3,1,0.5)
    #homework2()