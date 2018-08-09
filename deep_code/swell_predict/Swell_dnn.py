#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random as rn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# def swell_eval(y_true, y_pred):
#     print(y_true.info())
#     Score = 0
#     if y_true == 0:
#         if y_true == y_pred:
#             Score = Score+1
#         else:
#             Score = Score-1
#     else:
#         if y_true == y_pred:
#             Score = Score+2
#         else:
#             Score = Score-2
#     return K.sum(K.equal(y_true, K.round(y_pred)), axis=-1)


# fix random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
rn.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# manual_variable_initialization(True)
tf.global_variables_initializer()

test_dates = pd.read_csv('test_form.csv', usecols=[0], skiprows=[0, 1])
# test_dates = pd.read_csv(os.getcwd() +'\\deep_code\\dataset\\test_form.csv', usecols=[0], skiprows=[0, 1])
test_dates = test_dates.values.flatten().tolist()  # 제출해야할 날짜.



# 데이터 불러오기
X_df = pd.read_csv('.csv', index_col=[0])
X = X_df.values
Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y = Y_df.values  # 24시간 1001010111이 Y값
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)



number_of_var = len(X_train.columns)
first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)
print("first_layer_node_cnt %d" % first_layer_node_cnt)
# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(24))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

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
