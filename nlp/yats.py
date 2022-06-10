from keras.models import Model,load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import cv2
import pyaudio
import threading
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
import wave
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import librosa
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
FORMAT = pyaudio.paInt16
stats = ''
path = ''
yastarttag = 1


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.flatten = Flatten()
        self.f1 = Dense(128, activation='sigmoid')
        self.f2 = Dense(64, activation='sigmoid')
        self.f3 = Dense(32, activation='sigmoid')
        self.f4 = Dense(16, activation='sigmoid')
        self.f5 = Dense(3, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        y = self.f5(x)
        return y


class Stats:
    def __init__(self):
        self.ui = QUiLoader().load(r"E:\window\tiaoshi\pycharm\ya\GUI\mkf.ui")
        self.ui.pushButton.clicked.connect(self.yastart)
        self.ui.label.setStyleSheet('''
                     QLabel{
                     color:white;
                     font: bold;
                     font-size:30px;}
                ''')
        self.ui.pushButton.setStyleSheet('''
                         QPushButton{
                         color:black;
                         border:1px ridge pink;
                         background-color:

                            qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0.5 #faf0e6, stop: 0.8 #faf0e6,
                                    stop: 0.5 #faf0e6, stop: 1.0 #faf0e6);
                         }
                    ''')
        self.ui.groupBox.setStyleSheet('''
                                 QGroupBox{
                                 color:white;
                                 background-color:
                                    qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                    stop: 0.5 #ffb3e6, stop: 0.8 #faf0e6,
                                    stop: 0.5 #ffb3e6, stop: 1.0 #faf0e6);}  ## “#”号后面为色值！
                                 }
                            ''')
        self.ui.label.setText("请选择路径")

    def yastart(self):
        global yastarttag, path
        path = self.ui.lineEdit.text()  # 写
        self.ui.label.setText("系统正在运行")
        yastarttag = 0


def yaui():
    global stats
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    # print(shape)
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    # print(input.shape)
    # x = Flatten()(input)
    x = Dense(128, activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def accuracy(y_true, y_pred):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def yapre(y):
    ku = []
    str1 = "前进"
    str2 = '左转'
    str3 = '右转'
    str4 = '后退'
    for i in range(1, 4):
        yastr = path + r"E:\window\tiaoshi\pycharm\ya\nlp\traints\ima\s" + str(i) + r"\1.jpeg"
        x = cv2.imread(yastr)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, (28, 28))
        x = x.reshape((1, 784))
        x = x.astype('float32')
        x = x / 255
        p = [x, y]
        ku.append(model.predict(p)[0][0])
    ################################
    if (min(ku) >= 0.6):
        stats.ui.label.setText("预测结果:" + "无" + "(" + str(j) + ")")
    else:
        c = ku.index(min(ku))
        if (c == 0):
            yastr = 'go'
        elif (c == 1):
            yastr = 'left'
        else:
            yastr = 'right'
        stats.ui.label.setText("预测结果:" + yastr + "(" + str(j) + ")")


def yamkf():
    while (yastarttag):
        pass
    print(path)
    model.load_weights('my_model_weights.h5')
    count = 0
    CHUNK = 18000
    CHANNELS = 2
    RATE = 48000
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,  # 采样深度
                    channels=CHANNELS,  # 采样通道
                    rate=RATE,  # 采样率
                    input=True,
                    frames_per_buffer=CHUNK)  # 缓存
    print("开始缓存录音")
    frames = []
    j = 0
    while 1:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.short)
        temp = np.max(audio_data)
        if temp > 15000:
            print("检测到信号")
            j = j + 1
            data = stream.read(CHUNK)
            frames.append(data)
            WAVE_OUTPUT_FILENAME = "temp.wav"
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            y, sr = librosa.load('temp.wav', sr=20000)
            yapre(y)
            if (j >= 20):
                break
        else:
            # stats.ui.label.setText(str(count))
            count = count + 1
            if (count == 3):
                frames = []
                count = 0
    stream.stop_stream()
    stream.close()
    p.terminate()
    stats.ui.label.setText("运行完毕")


if __name__ == '__main__':
    input_shape = (784,)
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    rms = RMSprop()  # 优化器
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    threadLock = threading.Lock()
    threads = []
    thread1 = threading.Thread(target=yamkf)
    thread2 = threading.Thread(target=yaui)
    thread1.start()
    thread2.start()
    threads.append(thread1)
    threads.append(thread2)
    for t in threads:
        t.join()
    print("Exiting")
