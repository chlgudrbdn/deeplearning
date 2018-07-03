# import tensorflow as tf
#
# a = tf.random_uniform([2,3],seed=42)
#
# sess = tf.Session()
# sess.run( tf.global_variables_initializer() )
#
# for i in range(1):
#     print("a=", sess.run(a))

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_test.shape[0])
a = numpy.arange(7840000).reshape(10000, 28, 28, 1)
print(a)


# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255  #좀 독특해보이는 reshape인데
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
# Y_train = np_utils.to_categorical(Y_train)
# Y_test = np_utils.to_categorical(Y_test)