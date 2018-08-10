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
"""
def swell_eval(y_true, y_pred):
    print(y_true.info())
    Score = 0
    if y_true == 0:
        if y_true == y_pred:
            Score = Score+1
        else:
            Score = Score-1
    else:
        if y_true == y_pred:
            Score = Score+2
        else:
            Score = Score-2
    return K.sum(K.equal(y_true, K.round(y_pred)), axis=-1)
"""

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
# {'2015-07-18', '2015-06-27', '2014-12-21', '2014-09-25', '2016-03-04', '2015-04-04', '2014-07-06', '2014-05-18', '2014-10-23', '2016-10-20', '2015-12-13', '2015-01-13', '2017-03-15'}는 이 데이터 셋에선 test 못함.

# 데이터 불러오기
X_df = pd.read_csv('independent_var_with_Gu_and_Wall.csv', index_col=[0])

normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = set(normal_date).intersection(set(X_df.index.values))
abnormal_date = pd.read_csv('only_abnormal_date_data_without_swell.csv', index_col=[0]).values.flatten().tolist()
swell_date = pd.read_csv('only_swell_date_data.csv', index_col=[0]).values.flatten().tolist()
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))


normal_date_X_df = X_df.loc[normal_date].sample(len(swell_date))






X = X_df.values
Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y = Y_df.values  # 24시간 100101011... 같은 형태의 Y값
# X_train, X_test =
# Y_train =


'''
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









n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []

# 모델의 설정, 컴파일, 실행
for train, test in skf.split(X, Y): #이하 모델을 학습한 뒤 테스트. train과 test는 리스트.
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=100, batch_size=5)
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1]) # k fold가 항상 좋은 건 아니다. 만약 계층이 정렬되어 있으면 편중되게 추정할 가능성이 큼. 이걸 막으려고 scikit-learn에선 계층별로 또 k겹 교차 검증을 하는 방식을 사용. http://data-newbie.tistory.com/31참고
    accuracy.append(k_accuracy)

'''