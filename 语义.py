import cv2
import numpy as np
import random


if __name__ == '__main__':

    random.seed(10)
    #url = r'C:\Users\ASUS\Desktop\pic\3.jpeg'  # 根据摄像头设置IP及rtsp端口
    cap = cv2.VideoCapture(0)  # 读取视频流
    i=0
    j=0
    while (cap.isOpened()):
        x1 = int(random.random() * 255)
        x2 = int(random.random() * 255)
        x3 = int(random.random() * 255)
        ret, frame = cap.read()
        #print(frame)

        width = 1150
        height = 900
        cv2.namedWindow("ya",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ya", width, height)
        x = i
        if j==0:
            i=i+1
        else:
            i=i-1;
        if i==200:
            j=1
        if i==0:
            j=0
        x_center = 320
        y_center = 250
        imag = cv2.rectangle(frame, (x_center-x, y_center-x), (x_center+x, y_center+x), (x1, x2, x3), 2)
        imag = cv2.flip(imag, 1)

        cv2.imshow('ya', imag)
        boardkey = cv2.waitKey(1) & 0xFF
        if boardkey == 32:  # ascii
            break

    cap.release()
    cv2.destroyAllWindows()