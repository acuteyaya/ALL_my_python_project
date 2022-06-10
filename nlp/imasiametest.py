import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist
from keras.models import Model,load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import cv2


def euclidean_distance(vects):
 x, y = vects
 sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
 return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
 #print(shape)
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
 #x = Flatten()(input)
 x = Dense(128, activation='relu')(input)
 x = Dropout(0.1)(x)
 x = Dense(128, activation='relu')(x)
 x = Dropout(0.1)(x)
 x = Dense(128, activation='relu')(x)
 return Model(input, x)

def accuracy(y_true, y_pred): # Tensor上的操作
 '''Compute classification accuracy with a fixed threshold on distances.
 '''
 return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def yapre(y):
 ku=[]
 str1 = "chy"
 str2 = 'znj'
 str3 = 'hzh'
 for i in range(1,4):
  yastr=r"E:\window\tiaoshi\pycharm\ya\nlp\traints\ima\s"+str(i)+r"\1.jpeg"
  x = cv2.imread(yastr)
  x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
  x = cv2.resize(x,(28,28))
  x = x.reshape((1, 784))
  x = x.astype('float32')
  x = x / 255
  p=[x,y]
  print(len(p))
  ku.append(model.predict(p)[0][0])
 c=ku.index(min(ku))
 if(c==0):
  return str1
 elif(c==1):
  return str2
 else:
  return str3
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
 rms = RMSprop()#优化器
 model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
 model.load_weights('ya.h5')
 for i in range(1,6):
  yastr=r"E:\window\tiaoshi\pycharm\ya\nlp\traints\\"+str(i)+r".jpeg"
  y = cv2.imread(yastr)
  y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
  y = cv2.resize(y,(28,28))
  y = y.reshape((1, 784))
  y = y.astype('float32')
  y = y / 255
  print(yapre(y))
