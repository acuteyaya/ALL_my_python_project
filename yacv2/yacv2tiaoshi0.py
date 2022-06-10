import cv2
import numpy as np
from scipy import ndimage
def ya1():
    print("ya1:读入拼接")
    # 读入图片
    src = cv2.imread('watch.jpg',1)
    src1 = cv2.imread('watch.jpg',-1)

    # 调用cv.putText()添加文字
    text = "kb"
    AddText = src1.copy()
    cv2.putText(AddText, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (125, 55, 255), 3)

    # 将原图片和添加文字后的图片拼接起来
    res = np.hstack([src, AddText])

    # 显示拼接后的图片
    cv2.imshow('text', res)
    cv2.waitKey()
    cv2.destroyAllWindows()

def ya2():
    print("ya2:核")
    temp=-4
    kernel_3x3 = np.array([
        [temp, 2, temp],
        [2, 0, 2],
        [temp, 2, temp],
    ])

    kernel_5x5 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, 2, -1, -1],
        [-1, 2, 4, 2, -1],
        [-1, -1, 2, -1, -1],
        [-1, -1, -1, -1, -1],
    ])

    img = cv2.imread('watch.jpg', flags=cv2.IMREAD_GRAYSCALE)
    k3 = ndimage.convolve(img, kernel_3x3)
    k5 = ndimage.convolve(img, kernel_5x5)

    GBlur = cv2.GaussianBlur(img, (11, 11), 0)
    g_hpf = img - GBlur

    cv2.imshow('img', img)
    cv2.imshow('3x3', k3)
    cv2.imshow('5x5', k5)
    cv2.imshow('g_hpf', g_hpf)
    cv2.waitKey()
    cv2.destroyAllWindows()
def ya3():
    print('ya3:边缘')
    img = cv2.imread('watch.jpg', flags=cv2.IMREAD_GRAYSCALE)
    GBlur = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(GBlur, 50, 150)
    #cv2.imshow('img', img)
    cv2.imshow('canny', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def ya4():
    pass
if __name__ == '__main__':
    ya2()
