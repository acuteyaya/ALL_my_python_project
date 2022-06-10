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
stats=''
path=''
yastarttag=1

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
        global yastarttag,path
        path = self.ui.lineEdit.text()  # 写
        self.ui.label.setText("系统正在运行")
        yastarttag=0
def yaui():
    global stats
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
def yamkf():
    while (yastarttag):
        pass
    print(path)
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
            j=j+1
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
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)  # 20feature
            to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            data_set = []
            data_set.append([float(i) for i in to_append.split(" ")])
            x = []
            x.append(data_set[0])
            #print(x)
            model = LeNet5()
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                          metrics=['sparse_categorical_accuracy'])
            model.load_weights(path)
            y = model.predict(x)
            #print(np.max(y))
            if(np.max(y)<=0.6):
                stats.ui.label.setText("预测结果:" + "无" + "(" + str(j) + ")")
            else:
                r, c = np.where(y == np.max(y))
                if (c == 0):
                    yastr = 'go'
                elif (c == 1):
                    yastr = 'left'
                else:
                    yastr = 'right'
                stats.ui.label.setText("预测结果:"+yastr+"("+str(j)+")")
            if(j>=20):
                break
        else:
            #stats.ui.label.setText(str(count))
            count = count + 1
            if (count == 3):
                frames = []
                count = 0
    stream.stop_stream()
    stream.close()
    p.terminate()
    stats.ui.label.setText("运行完毕")
if __name__ == '__main__':
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