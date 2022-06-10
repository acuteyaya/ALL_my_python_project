import cv2
import sys

def CatchUsbVideo(window_name, camera_idx):
    num = 0
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    classfier = cv2.CascadeClassifier("E:\\window\\tiaoshi\\pycharm\\ya\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt2.xml")
    color = (255, 255, 255)
    while cap.isOpened():
        ok, frame = cap.read()
        if (not ok):
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))# 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                img_name = '%s\%d.jpg' % ("E:\\window\\tiaoshi\\pycharm\\ya\\ima\\zcl", num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                #print(image.shape)
                cv2.imwrite(img_name, image)
                num = num + 1
                if num > (1000):  # 如果超过指定最大保存数量退出循环
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 4)

        if num > (1000): break
        cv2.imshow(window_name, frame)
        boardkey = cv2.waitKey(1) & 0xFF
        if boardkey == 32:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    CatchUsbVideo("ya", r'http://192.168.31.192:4747/video')
