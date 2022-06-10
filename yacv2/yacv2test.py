import cv2
import sys
import random
import numpy as np
sys.path.append(r'E:\window\tiaoshi\pycharm\ya\yacv2')
from yacv2train1 import Model
MODEL_PATH = './model/zcl.face.model.h5'
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
cv2.ocl.setUseOpenCL(False)
x=y=w=h=0
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

if __name__ == '__main__':
    model = Model()
    model.load_model(file_path = MODEL_PATH)
    c1 = int(random.random() * 255)
    c2 = int(random.random() * 255)
    c3 = int(random.random() * 255)
    color = (c1, c2, c3)
    width = 700
    height = 900
    num = 0
    url1 = 'http://172.20.10.10:4747/video'
    url2 = 'http://192.168.31.192:4747/video'
    cap = cv2.VideoCapture(url2)
    cascade_path = r"E:\window\tiaoshi\pycharm\ya\yacv2\model\haarcascade_frontalface_alt2.xml"

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = rotate_bound(frame, 90)
        cv2.namedWindow("ya", 0)
        #cv2.resizeWindow("ya", width, height)
        cv2.rectangle(frame, (0 , 0 ), (476 + 1, 638 + 1), color, thickness=2)
        if (ret):
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)
                # 如果是“我”
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    # 文字提示是谁
                    cv2.putText(frame,'DAKEAI',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255 ,0 ,255),  # 颜色
                                2)  # 字的线宽
                else:
                    pass
        cv2.imshow("ya", frame)
        boardkey = cv2.waitKey(1) & 0xFF
        if boardkey == 32:  # ascii
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()