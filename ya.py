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
    # random.seed(10)
    x1 = int(random.random() * 255)
    x2 = int(random.random() * 255)
    x3 = int(random.random() * 255)
    width = 700
    height = 900
    lowThreshold = 48
    #url = 'http://192.168.1.107:4747/video'  # 根据摄像头设置IP及rtsp端口
    url = 'http://192.168.31.240:4747/video'
    cap = cv2.VideoCapture(url)  # 读取视频流

    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(frame)
        imag = rotate_bound(frame, 90)
        gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("ya", 0)
        cv2.resizeWindow("ya", width, height)
        #cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
        detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
        detected_edges = cv2.Canny(detected_edges,
                                   lowThreshold,
                                   lowThreshold * ratio,
                                   apertureSize=kernel_size)
        dst = cv2.bitwise_and(imag, imag, mask=detected_edges)  # just add some colours to edges from original image.
        cv2.imshow("ya", dst)

        boardkey = cv2.waitKey(1) & 0xFF
        if boardkey == 32:  # ascii
            break
    cap.release()
    cv2.destroyAllWindows()