import cv2
import numpy as np
import random

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

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
    c1 = int(random.random() * 255)
    c2 = int(random.random() * 255)
    c3 = int(random.random() * 255)
    color = (c1, c2, c3)
    width = 700
    height = 900
    num = 0
    #url = 'http://192.168.1.107:4747/video'  # 根据摄像头设置IP及rtsp端口
    url = 'http://192.168.31.170:4747/video'
    #url  = 'http://172.20.10.6:4747/video'
    #url = 'http://172.20.10.10:4747/video'

    path= 'E:\\window\\tiaoshi\\pycharm\\ya\\ima\\peigen'
    cap = cv2.VideoCapture(url)  # 读取视频流
    while (cap.isOpened()):
        ret, frame = cap.read()
        classfier = cv2.CascadeClassifier("E:\\window\\tiaoshi\\pycharm\\ya\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt2.xml")
        frame = rotate_bound(frame, 90)
        cv2.namedWindow("ya", 0)
        cv2.resizeWindow("ya", width, height)
        if (not ret):
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3,
                                               minSize=(32, 32))  # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                img_name = '%s\%d.jpg' % (path, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)
                num = num + 1
                if num > (1000):  # 如果超过指定最大保存数量退出循环
                    break
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        if num > (1000): break
        cv2.imshow("ya", frame)
        boardkey = cv2.waitKey(1) & 0xFF
        if boardkey == 32:  # ascii
            break
    cap.release()
    cv2.destroyAllWindows()