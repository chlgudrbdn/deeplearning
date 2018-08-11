#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import os, sys
import tensorflow as tf
import random as rn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
start_time = time.time()
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
X_df = pd.read_csv('independent_var_with_Wall.csv', index_col=[0])

X_df_index = set(X_df.index.values) - set(test_dates)  # 측정해야할 날짜는 뺀다.
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 측정일자와 데이터세트가 겹치는 날짜.
normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = set(normal_date).intersection(X_df_index)
abnormal_date = pd.read_csv('only_abnormal_date_data_without_swell.csv', index_col=[0]).values.flatten().tolist()
abnormal_date = set(abnormal_date).intersection(X_df_index)
swell_date = pd.read_csv('only_swell_date_data.csv', index_col=[0]).values.flatten().tolist()
swell_date = set(swell_date).intersection(X_df_index)
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))

# 날씨가 정상인날 1 : 날씨가 비정상인날(swell 제외) 1: swell이 일어나는 날 1 비율로 오버샘플링.
normal_date_X_df = X_df.loc[normal_date].sample(len(swell_date))
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([normal_date_X_df, abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)
X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values  # 24시간 100101011... 같은 형태의 Y값

number_of_var = len(X_train_df.columns)
first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)
print("first_layer_node_cnt %d" % first_layer_node_cnt)
epochs = 10
patience_num = 5
n_fold = 10
kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []
filename = os.path.basename(os.path.realpath(sys.argv[0]))

# 모델의 설정, 컴파일, 실행
for train_index, validation_index in kf.split(X):  # 이하 모델을 학습한 뒤 테스트.
    print("TRAIN:", train_index, "TEST:", validation_index)
    X_train, X_Validation = X[train_index], X[validation_index]
    Y_train, Y_Validation = Y[train_index], Y[validation_index]
    model = Sequential()
    model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu'))
    model.add(Dense(int(first_layer_node_cnt / 2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(int(first_layer_node_cnt / 4), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(int(first_layer_node_cnt / 8), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 판단근거 https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

    # 모델 저장 폴더 만들기
    # MODEL_DIR = './'+filename+' model_loopNum'+str(len(accuracy)).zfill(2)+'/'
    # if not os.path.exists(MODEL_DIR):
    #     os.mkdir(MODEL_DIR)
    # modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
    # # 모델 업데이트 및 저장
    # checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
    # 학습 자동 중단 설정
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=patience_num)

    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, verbose=2, callbacks=[early_stopping_callback])
    # history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, verbose=2, callbacks=[early_stopping_callback, checkpointer])

    plt.figure(figsize=(10, 10))
    # 테스트 셋의 오차
    y_vloss = history.history['val_loss']
    y_vacc = history.history['val_acc']
    y_loss = history.history['loss']
    y_acc = history.history['acc']
    # 그래프로 표현
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vacc, c="red", label='val_acc')
    plt.plot(x_len, y_acc, c="blue", label='acc')
    plt.plot(x_len, y_vloss, c="green", label='loss')
    plt.plot(x_len, y_loss, c="orange", label='val_loss')

    # 그래프에 그리드를 주고 레이블을 표시
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()

    Score = model.evaluate(X_Validation, Y_Validation)
    k_accuracy = "%.4f" % (Score[1])
    prediction_for_test = np.where(model.predict(X_Validation) <= 0.5, 0, 1)
    print("predict : ", prediction_for_test)
    print("real : ", Y_Validation)
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy:" % n_fold, accuracy)
accuracy = [float(j) for j in accuracy]
print("mean accuracy %.7f:" % np.mean(accuracy))

