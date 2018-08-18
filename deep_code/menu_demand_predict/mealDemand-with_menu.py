# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import numpy, os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rn
import pandas
# fix random seed for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(42)
rn.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# manual_variable_initialization(True)
tf.global_variables_initializer()

# load the dataset
filename = os.getcwd() + '\menu_preprocessed.csv'
# filename = os.getcwd() + '\dataset\menu_preprocessed.csv'
menu_df = pandas.read_csv(filename, index_col=[0], header=0)
menu_as_number_data = menu_df.values
menu_as_number_data = menu_as_number_data.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
menu_as_float_data = scaler.fit_transform(menu_as_number_data)

# 데이터 전처리
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_val = sequence.pad_sequences(x_test, maxlen=100)
'''
# 모델의 설정
model = Sequential()
model.add(Embedding(1878, 19))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu',strides=1)) # MNIST_Deep 에선 2차원 행렬 합성곱을 했지만 이경우는 1차원.
# padding: 바깥에 0을 채워넣냐 마냐.. "valid" 는 패딩 없단 소리. "same" 인풋과 같은 길이의 패딩 0 붙임(길이 조절은 불가). 결과적으로 출력 이미지 사이즈가 입력과 동일. "causal" 확대한 합성곱의 결과. 모델이 시간 순서를 위반해서는 안되는 시간 데이터를 모델링 할 때 유용.
# strides는 다음칸을 움직이는 칸수 정도로 보면 된다. 2이고 왼쪽에서 오른쪽 2칸 움직이고 다음 행으로 갈 땐 2칸 아래로 가는 식.
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.summary()

# 모델의 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델의 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=5, validation_data=(x_test, y_test))

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))


# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
'''