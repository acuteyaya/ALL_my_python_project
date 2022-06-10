import serial
from time import sleep
import cv2
import sys
import random
import numpy as np
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
import threading as thr
sys.path.append(r'E:\window\tiaoshi\pycharm\ya\yacv2')
from yacv2train1 import Model
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
cv2.ocl.setUseOpenCL(False)
x=y=w=h=0
start = 'ya'
datal = 'l'
datar = 'r'
datau = 'u'
datad = 'd'
yaall=1
yax=50
yay=50
yaxyc=2
bj=0
k=15
tag=1
usarttag=1
xtag=80
ytag=40
xtemp=80
ytemp=40

url=r'http://192.168.31.170:4747/video'
yamodelpath1='./model/zcl.face.model.h5'
yamodelpath2="E:\\window\\tiaoshi\\pycharm\\ya\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))
def send(str):
    #a = str+ "\n"
    # print(len(a))
    serial.write((str).encode("gbk"))
    sleep(0.1)
def receive():
    data = serial.read(10)
    t=data.decode("gbk").split("*")
    for i in t:
        if len(i)==3 and i.endswith('#'):
            return i[0:2]
class Stats:
    def __init__(self):
        self.ui = QUiLoader().load(r"E:\window\tiaoshi\pycharm\ya\GUI\stm32.ui")
        self.ui.pushButton.clicked.connect(self.changeyaall)
        self.ui.pushButton_3.clicked.connect(self.yausart)
        self.ui.pushButton_4.clicked.connect(self.yatag)
        self.ui.pushButton_14.clicked.connect(self.changeyaspeed)
        self.ui.pushButton_7.clicked.connect(self.addyax)
        self.ui.pushButton_6.clicked.connect(self.reduceyax)
        self.ui.pushButton_5.clicked.connect(self.addyay)
        self.ui.pushButton_2.clicked.connect(self.reduceyay)
        self.ui.pushButton_11.clicked.connect(self.changeyamodelpath1)
        self.ui.pushButton_12.clicked.connect(self.changeyamodelpath2)
        self.ui.pushButton_8.clicked.connect(self.changeyaurl)
        if (tag):
            self.ui.textEdit_6.setPlaceholderText("已打开")
        else:
            self.ui.textEdit_6.setPlaceholderText("已关闭")
        if (usarttag):
            self.ui.textEdit.setPlaceholderText("已打开")
        else:
            self.ui.textEdit.setPlaceholderText("已关闭")
        self.ui.textEdit_4.setPlaceholderText("X轴灵敏值:" + str(yax) + '\n'
                                             +"Y轴灵敏值:" + str(yay) )
    def changeyaall(self):
        global yaall
        threadLock.acquire()
        if(yaall==0):
            yaall = 1
        else:
            yaall = 0
        threadLock.release()
    def yausart(self):
        global usarttag
        threadLock.acquire()
        if (usarttag == 0):
            usarttag = 1
        else:
            usarttag = 0
        threadLock.release()
        if (usarttag == 1):
            self.ui.textEdit.setPlaceholderText("已打开")
        else:
            self.ui.textEdit.setPlaceholderText("已关闭")
    def yatag(self):
        global tag
        threadLock.acquire()
        if (tag == 0):
            tag = 1
        else:
            tag = 0
        threadLock.release()
        if (tag == 1):
            self.ui.textEdit_6.setPlaceholderText("已打开")
        else:
            self.ui.textEdit_6.setPlaceholderText("已关闭")
    def changeyaspeed(self):
        global k
        i1 = self.ui.lineEdit.text()  # 写
        i1=int(i1)
        if(i1>=0):
            threadLock.acquire()
            k = i1
            threadLock.release()
            self.ui.textEdit_3.setPlaceholderText("更改成功")
        else:
            self.ui.textEdit_3.setPlaceholderText("值错误")
    def changeyamodelpath1(self):
        global yamodelpath2
        i1 = self.ui.lineEdit_5.text()  # 写
        threadLock.acquire()
        yamodelpath2 = str(i1)
        threadLock.release()
        self.ui.textEdit_7.setPlaceholderText("导入成功")
    def changeyamodelpath2(self):
        global yamodelpath1
        i1 = self.ui.lineEdit_6.text()  # 写
        threadLock.acquire()
        yamodelpath1 = str(i1)
        threadLock.release()
        self.ui.textEdit_8.setPlaceholderText("导入成功")
    def addyax(self):
        global yax,yay,bj
        threadLock.acquire()
        yax=yax+yaxyc
        threadLock.release()
        self.ui.textEdit_4.setPlaceholderText("X轴灵敏值:" + str(yax) + '\n'
                                              + "Y轴灵敏值:" + str(yay) + '\n'
                                              + "        已刷新(" + str(bj) + ")")
        bj = bj + 1
        bj = bj % 99
    def reduceyax(self):
        global yax, yay, bj
        if(yax-yaxyc>=0):
            threadLock.acquire()
            yax = yax - yaxyc
            threadLock.release()
        self.ui.textEdit_4.setPlaceholderText("X轴灵敏值:" + str(yax) + '\n'
                                              + "Y轴灵敏值:" + str(yay) + '\n'
                                              + "        已刷新(" + str(bj) + ")")
        bj = bj + 1
        bj = bj % 99
    def addyay(self):
        global yax, yay, bj
        threadLock.acquire()
        yay = yay + yaxyc
        threadLock.release()
        self.ui.textEdit_4.setPlaceholderText("X轴灵敏值:" + str(yax) + '\n'
                                              + "Y轴灵敏值:" + str(yay) + '\n'
                                              + "        已刷新(" + str(bj) + ")")
        bj = bj + 1
        bj = bj % 99
    def reduceyay(self):
        global yax, yay, bj
        if (yay - yaxyc >= 0):
            threadLock.acquire()
            yay = yay - yaxyc
            threadLock.release()
        self.ui.textEdit_4.setPlaceholderText("X轴灵敏值:" + str(yax) + '\n'
                                              + "Y轴灵敏值:" + str(yay) + '\n'
                                              + "        已刷新(" + str(bj) + ")")
        bj = bj + 1
        bj = bj % 99
    def changeyaurl(self):
        global url
        i1 = self.ui.lineEdit_8.text()  # 写
        threadLock.acquire()
        url = str(i1)
        threadLock.release()
        self.ui.textEdit_5.setPlaceholderText("导入成功")
def yamain():
    global model,yaall,k,yax,yay,xtemp,ytemp
    while(yaall):
        pass
    print("start")
    c1 = int(random.random() * 255)
    c2 = int(random.random() * 255)
    c3 = int(random.random() * 255)
    color = (c1, c2, c3)
    width = 700
    height = 900
    midx = 238
    midy = 319
    k1 = 0
    cap = cv2.VideoCapture(url)
    # 人脸识别分类器本地存储路径
    cascade_path = yamodelpath2
    model.load_model(file_path=yamodelpath1)
    while (cap.isOpened()):

            ret, frame = cap.read()
            frame = rotate_bound(frame, 90)
            cv2.namedWindow("ya", 0)
            cv2.resizeWindow("ya", width, height)
            if(tag==1):
                cv2.rectangle(frame, (0 , 0 ), (476 , 638), color, thickness=1)
                cv2.rectangle(frame, (0 , 0 ), (238 , 319), color, thickness=1)
                cv2.rectangle(frame, (238, 319), (476, 638), color, thickness=1)
            if(k1==k):
                k1=0
                if (ret):
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    continue
                # 使用人脸识别分类器，读入分类器
                cascade = cv2.CascadeClassifier(cascade_path)

                # 利用分类器识别出哪个区域为人脸
                faceRects = cascade.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if len(faceRects) > 0:
                    for faceRect in faceRects:
                        x, y, w, h = faceRect
                        image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                        faceID = model.face_predict(image)
                        # 如果是“我”
                        #print("ok1")
                        if faceID == 0:
                            #print("ok2")
                            if (usarttag):
                                if (receive() == start):
                                    s1 = "*"
                                    s2 = "*"
                                    t1 = x + w // 2
                                    t2=  y + h // 2
                                    t1 = t1-midx
                                    t2 = t2-midy
                                    o1=70/midx * t1
                                    o1=int(float(yax/100)*o1)
                                    o2 = 40 / midy * t2
                                    o2 = int(float(yay / 100) * o2)
                                    kkk = xtemp + o1
                                    print(kkk)
                                    if(kkk>=0):
                                        xtemp = kkk
                                        o1 = xtag-xtemp
                                        if(o1 > 0):
                                            s1 = s1 + datal
                                        else:
                                            o1=o1*(-1)
                                            s1 = s1 + datar
                                    if(o1<10):
                                        s1 = s1 +"0"
                                    s1= s1 + str(o1)+"#"
                                    send(s1)

                                    kkk = ytemp + o2
                                    #print(kkk)
                                    if (kkk >= 0):
                                        ytemp = kkk
                                        o2 = ytag - ytemp
                                        if (o2 > 0):
                                            s2 = s2 + datau
                                        else:
                                            o2 = o2 * (-1)
                                            s2 = s2 + datad
                                    if (o2 < 10):
                                        s2 = s2 + "0"
                                    s2 = s2 + str(o2) + "#"
                                    send(s2)
                                    print()
                                #cv2.rectangle(frame, (x + w //2- 1, y  + h //2- 1), (x + w //2+1, y  + h //2+ 1), color, thickness=2)
                            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                            if(tag==1):
                                #print((x + w//2),(y + h//2))
                                #print()
                                cv2.rectangle(frame, (x - 10, y - 10), (x + w //2, y  + h //2), color, thickness=1)
                                cv2.rectangle(frame, (x + w//2,y + h//2),(x + w +10,y + h +10), color, thickness=1)
                            cv2.putText(frame, 'CUTEZCL',
                                        (x + 30, y + 30),  # 坐标
                                        cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                        1,  # 字号
                                        color,  # 颜色
                                        2)  # 字的线宽
                        else:
                            pass
            k1=k1+1
            cv2.imshow("ya", frame)
            boardkey = cv2.waitKey(1) & 0xFF
            if boardkey == 32:  # ascii
                break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
def yaui():
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
if __name__ == '__main__':
    model = Model()
    USART=1
    if(USART):
        serial = serial.Serial('COM19', 9600, timeout=2)  # /dev/ttyUSB0
    else:
        serial = serial.Serial('COM1', 9600, timeout=2)  # /dev/ttyUSB0
    if serial.isOpen():
        print("open success")
    else:
        print("open failed")
    threadLock = thr.Lock()
    threads = []
    thread = thr.Thread(target=yaui)
    thread.start()
    yamain()
    threads.append(thread)
    for t in threads:
        t.join()
    print("Exiting")
