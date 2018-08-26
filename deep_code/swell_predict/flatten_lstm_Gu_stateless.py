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
from datetime import datetime as dt
import os, sys
import tensorflow as tf
import random as rn
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

import time
start_time = time.time()


# convert series to supervised learning  # data는 dataframe을 의미
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):  # n_out은 후행지표가 있을 수도 있다고 판단해서인거 같다.
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
    return agg  # var1(t-1)...var8(t-1)   var1(t)...var8(t) 같은 형태로 출력됨.


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):  # 1이면 그냥 처음부터 끝의 한칸 전까지. 그 이상이면 . range(5)면 0~4 . 1031개 샘플 가진 데이터라면 look_back이 30일때 range가 1000. 즉 0~999=1000번 루프. 1을 빼야할 이유는 모르겠다.
        dataX.append(dataset[i:(i + look_back), ])  # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataY.append(dataset[i + look_back, -1])  # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return np.array(dataX), np.array(dataY)  # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦. 이짓을 대충 천번쯤 하는거다.


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


def divide_dateAndTime(df):
    onlyDate = []
    for dateAndtime in df.index.values.tolist():
        onlyDate.append(dt.strptime(dateAndtime, '%Y-%m-%d %H:%M').strftime('%Y-%m-%d'))
    # print(onlyDate[0])
    return set(onlyDate)


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

test_dates_df = pd.read_csv('test_dates_times.csv', index_col=[1], skiprows=0)  # 확인결과 중복있다.
test_dates_df = test_dates_df[~test_dates_df.index.duplicated(keep='first')]
test_dates_df.sort_index(inplace=True)  # 테스트할 데이터.
test_dates = test_dates_df.index.values.flatten().tolist()

# 데이터 불러오기
X_df = pd.read_csv('GuRyoungPo_hour.csv', index_col=[0])
X_df.sort_index(inplace=True)  # 데이터가 존재.

X_df_index = set(list(X_df.index.values)) - set(test_dates)  # 제출해야할 날짜는 우선적으로 뺀다.
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 예측해야할 날짜에 자료가 있는 시간.

abnormal_date = pd.read_csv('only_abnormal_not_swell_time_DF_flatten.csv', index_col=[0])
abnormal_date.sort_index(inplace=True)
abnormal_date = abnormal_date[abnormal_date['0'] == 1].index.values
abnormal_date = set(abnormal_date).intersection(X_df_index)

abnormal_only_date = divide_dateAndTime(X_df.loc[abnormal_date])  # 시간 빼고 날짜만 추출

swell_date = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
swell_date.sort_index(inplace=True)
swell_date = swell_date[swell_date['0'] == 1].index.values
swell_date = set(swell_date).intersection(X_df_index)

swell_only_date = divide_dateAndTime(X_df.loc[swell_date])  # 시간 빼고 날짜만 추출

# normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = (X_df_index - swell_date) - abnormal_date  # test도 swell도 비정상 날씨도 아닌 날.
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))


# 오버 샘플링 없이 모든 데이터 사용 : 그다지 예측력이 좋아진 느낌은 없다.
# normal_date_X_df = X_df.loc[normal_date]  # 일단 제외.
abnormal_date_X_df = X_df.loc[abnormal_date]  # 훈련에는 여기에 속한 날을 예측해야할 것이다.
swell_date_X_df = X_df.loc[swell_date]  # 훈련에는 여기에 속한 날을 예측해야할 것이다.

abnormal_and_swell_date_X_df = X_df.loc[abnormal_date, swell_date]
abnormal_and_swell_date_X_df.sort_index(inplace=True)
# X_train_df = X_df.loc[X_df_index]  # 일단 예측해야할 날짜의 데이터를 제외하고 가진 정보를 모두 발휘할 수 있는 범위

# X = X_train_df.values.astype('float32')
X = abnormal_and_swell_date_X_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_test = X_df.loc[test_dates_in_X_df]  # test_dates_in_X_df는 X_df_index와 겹치지 않는 부분. test하기 위한 기간의 정보.

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])  # 모든 시간이 완벽하게 갖춰져있다. 그러나 test를 위한 날짜도 포함되어 있다는 것을 기억해야함.
onlyXnotY = set(X_df.index.values) - set(Y_df.index.values)  # 여기엔 아무 값도 없어야 한다.
print(onlyXnotY)
onlyYnotX = set(Y_df.index.values) - set(X_df.index.values)  # 구룡포가 못다루는 시간대.
# print(len(onlyYnotX))
Y_df.sort_index(inplace=True)  # 혹시 몰라 정렬
Y_df_with_data = Y_df.loc[set(X_df.index.values)]  # 데이터가 있는 부분만 일단은 추출. test 하는 구간은 어차피 그걸 예측해야하니 생략
Y = Y_df_with_data.values

# hyperParameter
epochs = 300
patience_num = 200
n_hours = 4  # 일단 예측해야하는 날 사이 최소 11일 정도 간격 차이가 있다. 44개의 데이터로 어떻게든 학습하던가 아니면 보간된 걸로 어떻게든 해본다던가.
n_features = len(abnormal_and_swell_date_X_df.columns)  # 5
###############

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(X_df.values).astype('float32')
X_df_scaled = pd.DataFrame(data=scaled, index=X_df.index, columns=X_df.columns)
values_df = pd.concat([X_df_scaled, Y_df_with_data], axis=1, join='inner')  # 마지막에 Y값을 붙임. colum 명은 0이 됨.
values_df_outer = pd.concat([X_df_scaled, Y_df], axis=1, join='outer')  # 마지막에 Y값을 붙임. colum 명은 0이 됨. 당연하지만 빈곳 투성이일것.
# values = values_df.values


# frame as supervised learning
reframed = series_to_supervised(values_df.values, n_hours, 1)  # t+1 같은 데이터는 별로 필요 없어서 1로 n_out 지정
# reframed = series_to_supervised(scaled, n_hours, 1)  # t+1 같은 데이터는 별로 필요 없어서 1로 n_out 지정
reframed.drop(reframed.columns[-n_features:], axis=1, inplace=True)  # 예측해야하는건 이후의 모든 데이터가 아닌 1개의 데이터 뿐.

first_layer_node_cnt = int(reframed.shape[1]*(reframed.shape[1]-1)/2)  # 완전 연결 가정한 edge

# split into train and test sets
values = reframed.values
n_train_hours = (125 * 24) - n_hours  # 처음 예측해야하는 날짜 계산. test를 해보고 싶지만 일단 해본다. 나중에 for루프로 swell이 발생한 날과 아닌날을 regex이용해서 호출해내는 방식으로 훈련일을 정해야할 것 같다.
# print(values_df.iloc[n_hours + n_train_hours, :].index.name)  # 아마도 n_hour만큼 누락되었을거라 추측. 아무튼 이쪽

train = values[:n_train_hours, :]
test = values[n_train_hours: , :]  # 24시간만 예측할 수 있는 모델이면 족하다.
# split into input and outputs
# n_obs = n_hours * n_features  # swell 여부만 예측하면 그만이라 그다지 필요 없어 보인다.

# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
train_X, train_y = create_dataset(train, n_hours)
test_X, test_y = create_dataset(test, n_hours)

# print("train_X.shape, len(train_X), train_y.shape : %s" % train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, train_X.shape[2]))
test_X = test_X.reshape((test_X.shape[0], n_hours, test_X.shape[2]))
print("train_X.shape : %s \ntrain_y.shape : %s \ntest_X.shape : %s \ntest_y.shape : %s" % (train_X.shape, train_y.shape, test_X.shape, test_y.shape))

# 빈 accuracy 배열
accuracy = []
Scores = []
scriptName = os.path.basename(os.path.realpath(sys.argv[0]))

"""
for date in swell_only_date:
    forecast_date = values_df_outer.filter(regex=date, axis=0)
    
    print("----------------loop Num : ", num/3)
    # 최소 validation에 3일, tarin에 1일이라고 치면 가장 처음으로 빈칸이 생긴 날로부터 5일전이 최소로 필요. 물론 너무 적은 데이터라서 이 파트는 문제.
    # 일단 굴려보고 어떻게든 채워넣는 방법을 쓰는것도 나쁘진 않을 것 같다. !시간도 데이터도 부족하니 이 방법으로 간다.!
    # dt.strptime(str(test_only_date[num]), '%Y%m%d').date()는 최초로 빈칸이 생기는 날짜. +2 하면 제출일자가 된다.
    firstEmptyDate = dt.strptime(str(test_only_date[num]), '%Y%m%d').date()
    EndTrainDate = firstEmptyDate - timedelta(days=4)  # 20100709 # 2014-06-04 - 4 = 2010-05-30까지
    StartValidationDate = firstEmptyDate - timedelta(days=3)  # 20100710  # 20100705. 3일을 예측해야하지만 그전에 8일정도의 데이터를 기반으로 추측해서 RMSE를 추측할 예정.
    EndValidationDate = firstEmptyDate - timedelta(days=1)  # 20100712 : 20100705와는 8일차이.
    StartTestDate = firstEmptyDate  # 20100713
    EndTestDate = StartTestDate + timedelta(days=2)  # 20100715

    X_train_df = TrainXdf.loc[int(changeDateToStr(StartTrainDate)):int(changeDateToStr(EndValidationDate))]  # list slice와 달리 : 뒤쪽 항도 포함된다. # 일단 validataion 데이터도 같이 부른다.
    X_train = X_train_df.values
    # X_train = X_train_df.values[:, 22:]

    X_val_df = TrainXdf.loc[int(changeDateToStr(StartValidationDate)):int(changeDateToStr(EndValidationDate))]  # validation에 사용할 빈칸 이전의 3일관련 데이터.
    X_test_df = TestXdf.loc[int(changeDateToStr(StartTestDate)):int(changeDateToStr(EndTestDate))]
    blankCounta = X_test_df.shape[0]
    # if blankCounta != 12:
    #     print(X_val_df)

    # print("blankCounta : ", blankCounta)
    # print(X_train.shape)
    X_val = X_train[-look_back-blankCounta:, ]  # validation에 사용할 데이터 세트를 구축하기 위해 look_back 덧붙이기. 4*8 + 12
    X_train = X_train[:-blankCounta, ]  # validation할때 사용할 데이터만 제외하고 훈련

    print("StartTrainDate : ", StartTrainDate)
    print("EndTrainDate : ", EndTrainDate)
    # print(X_train.shape)
    print("StartValidationDate : ", StartValidationDate)
    print("EndValidationDate : ", EndValidationDate)
    # print(X_val.shape)
    print("StartTestDate : ", StartTestDate)
    print("EndTestDate : ", EndTestDate)

    X_train, y_train = create_dataset(X_train, look_back)
    X_val, y_val = create_dataset(X_val, look_back)
    # reshape training into [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], look_back, X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], look_back, X_val.shape[2])

    n_batch = gcd(X_train.shape[0], X_val.shape[0])  # 일단 배치사이즈를 대충 결정.
    model = Sequential()
    model.add(LSTM(number_of_var, batch_input_shape=(1, look_back, number_of_var), stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(number_of_var, batch_input_shape=(1, look_back, number_of_var), stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(number_of_var, batch_input_shape=(1, look_back, number_of_var), stateful=True))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 모델 저장 폴더 만들기
    MODEL_DIR = './' + scriptName + ' model_loopNum' + str(num).zfill(2) + '/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
    # # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
    # 학습 자동 중단 설정
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    custom_hist = CustomHistory()
    custom_hist.init()
    # fit network # 에포크 2로 해서 14바퀴 돌리니까 93분정도 걸림. 다 돌려면 332분 정도 들텐데 이거에 25배면 138시간 든다. 고작 epoch 50에 약 6일 소요. 시간부족. 단어는 무시하는 걸로 한정시키자. stateless로 한다던가 해야할듯.
    for i in range(epochs):
        model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False, validation_data=(X_val, y_val),
                  callbacks=[custom_hist, checkpointer])
        model.reset_states()

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName+' model_loopNum'+str(num).zfill(2))
    plt.plot(custom_hist.train_loss)
    plt.plot(custom_hist.val_loss)
    x_len = np.arange(len(custom_hist.val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    del model
    model = search_best_model(MODEL_DIR)
    evalScore = model.evaluate(X_val, y_val, batch_size=1)
    rmse_Scores.append(math.sqrt(evalScore))

    # 생각 같아선 추가된 데이터(validation 파트)를 포함해서 좀 더 훈련시켜두고 싶지만 마땅히 validation할 데이터가 없어 과적합 되기 십상. 그냥 이 모델 써서 예상해본다.
    for i in range(epochs):
        model.fit(X_val, y_val, epochs=1, batch_size=1, verbose=0, shuffle=False, validation_data=(X_train, y_train),
                  callbacks=[custom_hist, checkpointer])
        model.reset_states()

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName+' model_loopNum'+str(num).zfill(2))
    plt.plot(custom_hist.train_loss)
    plt.plot(custom_hist.val_loss)
    plt.ylim(0.0, 50.0)
    x_len = np.arange(len(custom_hist.val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    del model
    model = search_best_model(MODEL_DIR)

    X_test = X_test_df.values
    x_hat = X_train[-1:, ]  # 끄트머리에서 다른 변수들로 계산.
    # x_hat = X_train[-look_back:, ]  # 끄트머리에서 다른 변수들로 계산.
    testPredict = np.zeros((blankCounta, 1))
    test_start_area_absolute_position = TrainXdf.index.get_loc(int(changeDateToStr(StartTestDate))).start
    for k in range(blankCounta):
        prediction = model.predict(x_hat, batch_size=n_batch)
        # prediction = model.predict(np.array([x_hat]), batch_size=1)
        testPredict[k] = prediction
        # new_x_hat = np.append([X_test[k, ], prediction])
        new_x_hat = np.vstack([np.reshape(X_test[k], (-1, 1)), prediction])
        x_hat = np.asarray([np.vstack([x_hat[0][1:], np.reshape(new_x_hat, (1, -1))])])
        # print(x_hat.shape)
        TrainXdf.iloc[test_start_area_absolute_position + k, -1] = prediction

    forecast_satck.extend(testPredict)
    # if num != len(test_only_date) - 3: # 마지막 루프만 아니면
    #     StartTrainDate = EndTestDate + timedelta(days=1)  # 다음 루프때 쓸 train data 구간 규정
    m, s = divmod((time.time() - start_time), 60)
    print("almost %d minute" % m)
    # break





























# design network
model = Sequential()
model.add(LSTM(first_layer_node_cnt, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=72, validation_data=(test_X, test_y), verbose=2)
# plot history
plt.figure(figsize=(8, 8))
# 테스트 셋의 오차
y_acc = history.history['acc']
y_vacc = history.history['val_acc']
y_loss = history.history['loss']
y_vloss = history.history['val_loss']
# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_acc, c="blue", label='acc')
plt.plot(x_len, y_vacc, c="red", label='val_acc')
plt.plot(x_len, y_loss, c="green", label='loss')
plt.plot(x_len, y_vloss, c="orange", label='val_loss')

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
score = score_calculating(inv_y, inv_yhat)
print('Test score: %.3f' % score)
"""
