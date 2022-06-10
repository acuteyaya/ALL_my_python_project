import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader

from PySide2.QtGui import QImage, QPixmap

checkpoint_save_path = "./checkpoint/Baseline.ckpt"
path1 = "./pic/1.jpg"

class Stats:

    def __init__(self):
        self.ui = QUiLoader().load("ya.ui")

        self.ui.pushButton_2.clicked.connect(self.handleCalc1)#模型

        self.ui.pushButton_4.clicked.connect(self.handleCalc2)#检测

    def handleCalc1(self):
        global checkpoint_save_path,path1
        i1 = self.ui.lineEdit.text()#图片
        i2 = self.ui.lineEdit_2.text()  #模型
        checkpoint_save_path = i2
        path1 = i1

        #checkpoint_save_path = "./checkpoint/Baseline.ckpt"

    def handleCalc2(self):
        if os.path.exists(checkpoint_save_path + '.index'):
            model.load_weights(checkpoint_save_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,
                                                         save_best_only=True)
        image = cv2.imread(path1)
        image = cv2.resize(image, (880, 400))
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        stats.ui.label.setPixmap(QPixmap.fromImage(image))

        img1 = cv2.imread(path1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img1 = cv2.resize(img1, (28, 28), interpolation=cv2.INTER_CUBIC)
        # print(img1)
        # plt.imshow(img1)
        # plt.show()
        img1 = tf.cast(img1, tf.float32)
        img1 = img1 / 255
        img1 = np.array(img1)
        img1 = np.expand_dims(img1, axis=0)
        wz = model.predict(img1, batch_size=64 ,callbacks=[cp_callback])
        print("type", wz.shape)
        print("wz:",wz)
        pd(wz)
        self.ui.textEdit.setPlaceholderText(pd(wz))

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation('relu')  # 激活层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d2 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')  # 卷积层
        self.b3 = BatchNormalization()  # BN层
        self.a3 = Activation('relu')  # 激活层
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d3 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f5 = Dense(288, activation='relu')
        self.d5 = Dropout(0.2)
        self.f6 = Dense(72, activation='relu')
        self.d6 = Dropout(0.2)
        self.f7 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.flatten(x)

        x = self.f5(x)
        x = self.d5(x)

        x = self.f6(x)
        x = self.d6(x)

        y = self.f7(x)
        return y


def pd(k):
    biao = ""
    if k[0][0:1] == 1:
        biao = "airplane"
    elif k[0][1:2] == 1:
        biao = "automobile"
    elif k[0][2:3] == 1:
        biao = "bird"
    elif k[0][3:4] == 1:
        biao = "cat"
    elif k[0][4:5] == 1:
        biao = "deer"
    elif k[0][5:6] == 1:
        biao = "dog"
    elif k[0][6:7] == 1:
        biao = "frog"
    elif k[0][7:8] == 1:
        biao = "horse"
    elif k[0][8:9] == 1:
        biao = "3"
    elif k[0][9:10] == 1:
        biao = "truck"
    print(biao)
    return biao

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    model = Baseline()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
