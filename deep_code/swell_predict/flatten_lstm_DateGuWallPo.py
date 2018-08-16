#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM
from keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import os, sys
import tensorflow as tf
import random as rn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
start_time = time.time()

def weight_variable(shape, name=None):
    return np.sqrt(0.01 / shape[0]) * np.random.normal(size=shape)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def score_calculating(true_value, pred_value):
    Score = 0
    for i in range(len(true_value)):
        for j in range(len(true_value[i])):
            if true_value[i][j] == 0:
                if true_value[i][j] == pred_value[i][j]:
                    Score = Score + 1
                else:
                    Score = Score - 1
            else:
                # print(true_value[i][j], ":", pred_value[i][j], end=",")
                if true_value[i][j] == pred_value[i][j]:
                    Score = Score + 2
                else:
                    Score = Score - 2
    return Score


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

test_dates_df = pd.read_csv('test_dates_times.csv', usecols=[1])
test_dates = test_dates_df.values.flatten().tolist()
# 데이터 불러오기
# X_df = pd.read_csv('ind_var_with_DateGuWallPo.csv', index_col=[0])
X_df = pd.read_csv('ind_var_with_DateGuWallPo_withoutwind.csv', index_col=[0])

X_df_index = set(list(X_df.index.values)) - set(test_dates)  # 제출해야할 날짜는 뺀다.
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 측정일자와 데이터세트가 겹치는 시간.
abnormal_date = pd.read_csv('only_abnormal_not_swell_time_DF_flatten.csv', index_col=[0])
abnormal_date = abnormal_date[abnormal_date['0'] == 1].index.values
abnormal_date = set(abnormal_date).intersection(X_df_index)
swell_date = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
swell_date = swell_date[swell_date['0'] == 1].index.values
swell_date = set(swell_date).intersection(X_df_index)
# normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = (X_df_index-swell_date) - abnormal_date  # test도 swell도 비정상 날씨도 아닌 날.
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))


# 오버 샘플링 없이 모든 데이터 사용 : 그다지 예측력이 좋아진 느낌은 없다.
'''
# normal_date_X_df = X_df.loc[normal_date]
# abnormal_date_X_df = X_df.loc[abnormal_date]
# swell_date_X_df = X_df.loc[swell_date]
X_train_df = X_df.loc[X_df_index]

X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)

X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 비정상인날(swell제외) 전부 : swell이 일어나는 날. 대회에선 정상인 날자는 신경쓰지 않는다고 첫날에 가정. : 그다지 예측력이 좋아진 느낌은 없다.

abnormal_date_X_df = X_df.loc[abnormal_date]
swell_date_X_df = X_df.loc[swell_date]

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)

X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values

# 날씨가 비정상인날(swell제외) 1 : swell이 일어나는 날 1 비율로 오버 샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)

X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 정상인날 1 : 날씨가 비정상인날(swell 제외) 1: swell이 일어나는 날 1 비율로 오버샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
normal_date_X_df = X_df.loc[normal_date].sample(len(swell_date))
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([normal_date_X_df, abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)

X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# swell로만 학습 : 형편없다. 나중에 결과를 합치는데 써봐야할 것이다.
'''
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)
X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values  # 24시간 100101011... 같은 형태의 Y값
'''

n_hours = 3
# number_of_var = len(X_train_df.columns)
n_features = len(X_train_df.columns)
first_layer_node_cnt = int(n_features*(n_features-1)/2)
print("first_layer_node_cnt %d" % first_layer_node_cnt)
epochs = 300
patience_num = 200
# n_fold = 10
# kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []
Scores = []
scriptName = os.path.basename(os.path.realpath(sys.argv[0]))

# specify the number of lag hours
# frame as supervised learning
reframed = series_to_supervised(X, n_hours, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invert scaling for forecast
inv_yhat = pd.concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = X_scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


'''
# 모델의 설정, 컴파일, 실행
for train_index, validation_index in kf.split(X):  # 이하 모델을 학습한 뒤 테스트.
    print("loop num : ", len(accuracy)+1)
    print("TRAIN: %d" % len(train_index), "TEST: %d" % len(validation_index))

    X_train, X_Validation = X[train_index], X[validation_index]
    Y_train, Y_Validation = Y[train_index], Y[validation_index]
    model = Sequential()
    model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu', kernel_initializer='random_normal'))
    edge_num = 2
    # model.add(Dense(int(first_layer_node_cnt * (edge_num**(-2))), activation='relu'))
    while int(first_layer_node_cnt * (edge_num**(-2))) >= 5 and edge_num < 6:
        model.add(Dense(int(first_layer_node_cnt * (edge_num**(-2))), kernel_initializer='random_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dropout(0.1))
        edge_num += 1
    model.add(Dense(1, activation='sigmoid'))
    print("edge_num : %d" % edge_num)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 판단근거 https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])  # 판단근거 https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

    # 모델 저장 폴더 만들기
    MODEL_DIR = './'+scriptName+' model_loopNum'+str(len(accuracy)).zfill(2)+'/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
    # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
    # 학습 자동 중단 설정
    # early_stopping_callback = EarlyStopping(monitor='val_acc', patience=patience_num)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)

    history = model.fit(X_train, Y_train, validation_data=(X_Validation, Y_Validation), epochs=epochs, verbose=0, batch_size=len(X_train),
                        callbacks=[early_stopping_callback, checkpointer])
    # history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, verbose=2, callbacks=[early_stopping_callback, checkpointer])

    plt.figure(figsize=(8, 8))
    # 테스트 셋의 오차
    # y_acc = history.history['binary_accuracy']
    y_acc = history.history['acc']
    # y_vacc = history.history['val_binary_accuracy']
    y_vacc = history.history['val_acc']
    y_loss = history.history['loss']
    y_vloss = history.history['val_loss']
    # 그래프로 표현
    x_len = np.arange(len(y_loss))
    # plt.plot(x_len, y_acc, c="blue", label='binary_accuracy')
    plt.plot(x_len, y_acc, c="blue", label='acc')
    # plt.plot(x_len, y_vacc, c="red", label='val_binary_accuracy')
    plt.plot(x_len, y_vacc, c="red", label='val_acc')
    plt.plot(x_len, y_loss, c="green", label='loss')
    plt.plot(x_len, y_vloss, c="orange", label='val_loss')

    # 그래프에 그리드를 주고 레이블을 표시
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()

    file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
    file_list.sort()
    print(file_list[0])
    model = load_model(MODEL_DIR + file_list[0])

    Score = model.evaluate(X_Validation, Y_Validation, batch_size=len(X_Validation))
    k_accuracy = "%.4f" % (Score[1])
    prediction_for_test = np.where(model.predict(X_Validation) < 0.5, 0, 1)
    print(prediction_for_test.sum())
    # print("predict : %s" % prediction_for_test)
    # print("real    : %s" % Y_Validation)
    Scores.append(score_calculating(Y_Validation, prediction_for_test))
    print("\nscore guess : %d" % score_calculating(Y_Validation, prediction_for_test))
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy:" % n_fold, accuracy)
accuracy = [float(j) for j in accuracy]
print("mean accuracy %.7f:" % np.mean(accuracy))
print("score : %s" % Scores)
print("mean score : %.4f" % np.mean(Scores))

print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %2f minute" % m)


model = Sequential()
model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu', kernel_initializer='random_normal'))
edge_num = 2
# model.add(Dense(int(first_layer_node_cnt * (edge_num**(-2))), activation='relu'))
while int(first_layer_node_cnt * (edge_num ** (-2))) >= 5 and edge_num < 6:
    model.add(Dense(int(first_layer_node_cnt * (edge_num ** (-2))), kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    edge_num += 1
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

MODEL_DIR = './'+scriptName+' model_loopNum'+str(len(accuracy)).zfill(2)+'/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)

history = model.fit(X, Y, validation_split=0.1, epochs=epochs, verbose=0, batch_size=len(X),
                    callbacks=[checkpointer, early_stopping_callback])

plt.figure(figsize=(8, 8))
# 테스트 셋의 오차
# y_acc = history.history['binary_accuracy']
y_acc = history.history['acc']
# y_vacc = history.history['val_binary_accuracy']
y_vacc = history.history['val_acc']
y_loss = history.history['loss']
y_vloss = history.history['val_loss']
# 그래프로 표현
x_len = np.arange(len(y_loss))
# plt.plot(x_len, y_acc, c="blue", label='binary_accuracy')
plt.plot(x_len, y_acc, c="blue", label='acc')
# plt.plot(x_len, y_vacc, c="red", label='val_binary_accuracy')
plt.plot(x_len, y_vacc, c="red", label='val_acc')
plt.plot(x_len, y_loss, c="green", label='loss')
plt.plot(x_len, y_vloss, c="orange", label='val_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper left')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
file_list.sort()  # 만든날짜 정렬
print(file_list[0])
model = load_model(MODEL_DIR + file_list[0])

prediction_for_test = np.where(model.predict(X_test.values, batch_size=len(X_test)) < 0.5, 0, 1)
# for timeAndDate, predic in zip(X_test.index.values, prediction_for_test):
#     print("%s" % timeAndDate, ": %d" % predic)
prediction_for_test_DF_DateGuWall = pd.DataFrame(data=prediction_for_test, index=X_test.index.values)

# print(prediction_for_test_DF_DateGuWall)
print(prediction_for_test_DF_DateGuWall.sum())
print(prediction_for_test_DF_DateGuWall.shape)
prediction_for_test_DF_DateGuWall.to_csv('prediction_for_test_DF_DateGuWall.csv', encoding='utf-8')
'''
