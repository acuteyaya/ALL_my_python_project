'''
File Name          : homework.c
Author             : CUTEYAYA
Version            : V1.0.0
Created on         : 2021/12/15
'''

import requests      # 发送网络请求模块
import json
import xlrd
import xlwt
import matplotlib.pyplot as plt
import numpy as np
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap
import cv2

ya11 = []
ya12 = []
ya21 = []
ya22 = []
timelist = []
path1=''
path2=''
bj=1

def yawxls():
    n = requests.get(url="https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5")
    m = json.loads(n.text)
    data = json.loads(m['data'])

    list = [  ['截至时间',str(data['lastUpdateTime'])],
              ['全国确诊人数',str(data['chinaTotal']['confirm'])],
              ['全国疑似', str(data['chinaTotal']['suspect'])],
              ['全国治愈', str(data['chinaTotal']['heal'])],
              ['全国死亡', str(data['chinaTotal']['dead'])],
              ['今日新增确诊', str(data['chinaAdd']['confirm'])],
              ['今日新增疑似',str(data['chinaAdd']['suspect'])],
              ['今日新增治愈',str(data['chinaAdd']['heal'])],
              ['今日新增死亡',str(data['chinaAdd']['dead'])]]
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('ya', cell_overwrite_ok=True)
    for i in range(len(list)):
        for j in range(len(list[i])):
            worksheet.write(i, j, label=str(list[i][j]))
    workbook.save(path1)

def yarxls():
    global ya11,ya12,ya21,ya22,timelist
    ya11 = []
    ya12 = []
    ya21 = []
    ya22 = []
    timelist = []
    data = xlrd.open_workbook(path2)
    table = data.sheet_by_name(u'ya')
    time = table.row_values(0)
    for i in range(0, 2):
        timelist.append(time[i])
    for i in range(1,5):
        rows = table.row_values(i)
        ya11.append(rows[0])
        ya12.append(int(rows[1]))
    for i in range(5, 9):
        rows = table.row_values(i)
        ya21.append(rows[0])
        ya22.append(int(rows[1]))

def set_label(rects):
    for rect in rects:
        height = rect.get_height()  # 获取?度
        plt.text(x=rect.get_x() + rect.get_width() / 2,  # ?平坐标
                 y=height + 0.5,  # 竖直坐标
                 s=height,  # ?本
                 ha='center')  # ?平居中
def draw1():
    labels = ya11  # 级别
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 6))
    plt.xticks(x, labels)
    width = 0.3
    rects1 = plt.bar(x , ya12, width)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('实时全国疫情情况          '+timelist[0]+timelist[1])
    plt.ylim((0, 150000))
    plt.ylabel("人数")
    set_label(rects1)
    plt.tight_layout()  # 设置紧凑布局
    plt.savefig('test1.jpg')
    #plt.show()

# stats.ui.label.setPixmap(QPixmap.fromImage(image))
def draw2():
    labels = ya21  # 级别
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 6))
    plt.xticks(x, labels)
    width = 0.3
    rects1 = plt.bar(x , ya22, width)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('实时今日疫情情况        '+timelist[0]+timelist[1])
    plt.ylim((0, 300))
    plt.ylabel("人数")
    set_label(rects1)
    plt.tight_layout()  # 设置紧凑布局
    plt.savefig('test2.jpg')
    #plt.show()

class Stats:
    def __init__(self):
        self.ui = QUiLoader().load("./ya.ui")
        self.ui.pushButton.clicked.connect(self.handleCalc1)#爬虫
        self.ui.pushButton_2.clicked.connect(self.handleCalc2)#读取
        self.ui.pushButton_3.clicked.connect(self.handleCalc3)#全国
        self.ui.pushButton_4.clicked.connect(self.handleCalc4)#今日
        self.ui.pushButton_5.clicked.connect(self.handleCalc5)#一键爬虫
    def handleCalc1(self):
        global path1
        i1 = self.ui.lineEdit.text()#写
        path1=i1
        yawxls()
    def handleCalc2(self):
        global path2,bj
        i2 = self.ui.lineEdit_2.text()  # 读
        path2 = i2
        yarxls()
        self.ui.textEdit_2.setPlaceholderText("已刷新("+str(bj)+")")
        bj=bj+1
        bj=bj%99
    def handleCalc3(self):
        draw1()
        image = cv2.imread('test1.jpg')
        image = cv2.resize(image, (700, 400))
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        stats.ui.label.setPixmap(QPixmap.fromImage(image))
    def handleCalc4(self):
        draw2()
        image = cv2.imread('test2.jpg')
        image = cv2.resize(image,(700, 400))
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        stats.ui.label.setPixmap(QPixmap.fromImage(image))
    def handleCalc5(self):
        global path1,path2,bj
        i1 = self.ui.lineEdit.text()  # 写
        i2 = self.ui.lineEdit_2.text()  # 读
        path1 = i1
        path2 = i2
        yawxls()
        yarxls()
        self.ui.textEdit_2.setPlaceholderText("已刷新(" + str(bj) + ")")
        bj=bj+1
        bj=bj%99

if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
