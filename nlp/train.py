import os
import random
import wave
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import librosa
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
np.set_printoptions(threshold=np.inf)
def yaload(path_name):
    x_all=[]
    y_all=[]
    k=0
    for Dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, Dir_item))
        features = []
        for dir_item in os.listdir(full_path):
            y, sr = librosa.load(full_path+"\\"+dir_item, sr=20000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            features.append(mfcc)



        samples = random.sample(features, 100)
        samples = np.vstack(samples)
        # 平均MFCC的值为了归一化处理
        mfcc_mean = np.mean(samples, axis=0)
        # 计算标准差为了归一化
        mfcc_std = np.std(samples, axis=0)
        print(mfcc_mean)
        print(mfcc_std)
        # 归一化特征
        features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]
        print(len(features), features[0].shape)

        #x_all.append(data_set[0])
        #y_all.append(k)

        print(k)
        k=k+1
    return x_all,y_all

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.flatten = Flatten()

        self.f1 = Dense(600, activation='relu')
        self.d6 = BatchNormalization()
        self.f2 = Dense(600, activation='relu')
        self.d7 = BatchNormalization()
        self.f3 = Dense(400, activation='relu')
        self.d8 = Dropout(0.5)
        self.f4 = Dense(300, activation='relu')
        self.d9 = Dropout(0.5)
        self.f5 = Dense(4, activation='softmax')

    def call(self, x):

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        x = self.f3(x)
        x = self.d8(x)
        x = self.f4(x)
        x = self.d9(x)
        y = self.f5(x)
        return y

if __name__ == '__main__':
    if(1):
        x_all,y_all=yaload(r"E:\window\tiaoshi\pycharm\ya\nlp\train")
        x_all = np.array(x_all, dtype=np.float64)
        y_all = np.array(y_all, dtype=np.float64)
        np.save('x.npy', x_all)
        np.save('y.npy', y_all)
    x_all = np.load('x.npy')
    y_all = np.load('y.npy')


    x_train,x_test,y_train,y_test =  train_test_split(x_all, y_all, test_size=0.2,
                                                                              random_state=random.randint(0, 100))
    #x_test=np.absolute(x_test)
   # x_train = np.absolute(x_train)
    print(x_train.shape)
    print(x_test.shape)

    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(
    #     x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 19)
    # x_test = scaler.transform(
    #     x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 19)
    # x_train=x_train+1
    # x_test=x_test+1

    #平均MFCC的值为了归一化处理
    np.random.seed(116)
    np.random.shuffle(x_train)
    np.random.seed(116)
    np.random.shuffle(y_train)
    tf.random.set_seed(116)
    #print(len(x_train))
    #print(len(y_train))
    #print(len(x_test))
    #print(len(y_test))
    model = LeNet5()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    if not os.path.exists(r"./model"):
        os.makedirs(r"./model")
    checkpoint_save_path = "./model/rnn_embedding_4pre1.ckpt"

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    else:
        print('-------------create the model-----------------')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=1,
                                                    )
    history = model.fit(x_train, y_train, batch_size=64, epochs=500, validation_data=(x_test, y_test), validation_freq=1,callbacks=[cp_callback])
    model.summary()
    #model.save_weights(checkpoint_save_path+".index")
    ###############################################    show   ###############################################
    #model.save_weights('my_model_weights.h5')
    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig("ya.jpg")
    plt.show()