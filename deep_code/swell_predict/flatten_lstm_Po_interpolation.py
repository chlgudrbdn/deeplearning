# -*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from datetime import datetime as dt
from datetime import timedelta
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
    for i in range(len(
            dataset) - look_back):  # 1이면 그냥 처음부터 끝의 한칸 전까지. 그 이상이면 . range(5)면 0~4 . 1031개 샘플 가진 데이터라면 look_back이 30일때 range가 1000. 즉 0~999=1000번 루프. 1을 빼야할 이유는 모르겠다.
        dataX.append(dataset[i:(i + look_back), ])  # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataY.append(dataset[i + look_back,])  # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return np.array(dataX), np.array(dataY)  # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦. 이짓을 대충 천번쯤 하는거다.


'''
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
'''

def divide_dateAndTime(df):
    onlyDate = []
    for dateAndtime in df.index.values.tolist():
        onlyDate.append(dt.strptime(dateAndtime, '%Y-%m-%d %H:%M').strftime('%Y-%m-%d'))
    # print(onlyDate[0])
    return set(onlyDate)


def search_best_model(MODEL_DIR):
    file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
    file_list = [float(fileName[:-5]) for fileName in file_list]
    file_list.sort()  # 만든날짜 정렬
    model = load_model(MODEL_DIR + '{0:.9f}'.format(file_list[0]) + ".hdf5")
    return model


class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def gcd(a, b):
    if a < b:
        (a, b) = (b, a)
    while b != 0:
        (a, b) = (b, a % b)
    return a


# 출처: http://codepractice.tistory.com/65 [코딩 연습]


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
X_df = pd.read_csv('Pohang_hour.csv', index_col=[0])
# X_df = pd.read_csv('Pohang_hour_without_wind.csv', index_col=[0])
X_df.sort_index(inplace=True)  # 데이터가 존재.

X_df_index = set(list(X_df.index.values)) - set(test_dates)  # 제출해야할 날짜를 제외한 부분은 우선 확인
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 제출해야할 날짜에 자료가 있는 시간.

# 인덱스 호출시 사용할
# abnormal_date = pd.read_csv('only_abnormal_not_swell_time_DF_flatten.csv', index_col=[0])
# abnormal_date.sort_index(inplace=True)
# abnormal_date = abnormal_date[abnormal_date['0'] == 1].index.values
# abnormal_date = set(abnormal_date).intersection(X_df_index)

# abnormal_only_date = divide_dateAndTime(X_df.loc[abnormal_date])  # 시간 빼고 날짜만 추출

# swell_date = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
# swell_date.sort_index(inplace=True)
# swell_date = swell_date[swell_date['0'] == 1].index.values
# swell_date = set(swell_date).intersection(X_df_index)

# swell_only_date = divide_dateAndTime(X_df.loc[swell_date])  # 시간 빼고 날짜만 추출

# normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
# normal_date = (X_df_index - swell_date) - abnormal_date  # test도 swell도 비정상 날씨도 아닌 날.
# print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])  # 모든 시간이 완벽하게 갖춰져있다. 그러나 test를 위한 날짜도 포함되어 있다는 것을 기억해야함.
onlyXnotY = set(X_df.index.values) - set(Y_df.index.values)  # 여기엔 아무 값도 없어야 한다.
print(onlyXnotY)
onlyYnotX = set(Y_df.index.values) - set(X_df.index.values)
# print(len(onlyYnotX))
Y_df.sort_index(inplace=True)  # 혹시 몰라 정렬
# Y_df_with_data = Y_df.loc[set(X_df.index.values)]  # 데이터가 있는 부분만 일단은 추출. test 하는 구간은 어차피 그걸 예측해야하니 생략
# Y = Y_df_with_data.values

values_df_outer = pd.concat([X_df, Y_df], axis=1, join='outer', sort=True)
values_df_outer = values_df_outer.drop(columns=['0'])  # 그 자체로 불안정하기 때문에(일단 test date에도 swell이 안일어난 것으로 해뒀음) 실제 데이터 분석하기 전엔 안 쓸 예정.
values_df_outer = values_df_outer.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))

# hyperParameter
# epochs = 2
epochs = 100
# patience_num = 2
# patience_num = 20
n_hours = 4  # 일단 예측해야하는 날 사이 최소 11일 정도 간격 차이가 있다. 44개의 데이터로 어떻게든 학습하던가 아니면 보간된 걸로 어떻게든 해본다던가.
n_features = len(values_df_outer.columns)  #
# n_obs = n_hours * n_features  # swell 여부만 예측하면 그만이라 그다지 필요 없어 보인다.
###############

'''
X_test = X_df.loc[test_dates_in_X_df]  # test_dates_in_X_df는 X_df_index와 겹치지 않는 부분. test하기 위한 기간의 정보.

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])  # 모든 시간이 완벽하게 갖춰져있다. 그러나 test를 위한 날짜도 포함되어 있다는 것을 기억해야함.
onlyXnotY = set(X_df.index.values) - set(Y_df.index.values)  # 여기엔 아무 값도 없어야 한다.
print(onlyXnotY)
onlyYnotX = set(Y_df.index.values) - set(X_df.index.values)
# print(len(onlyYnotX))
Y_df.sort_index(inplace=True)  # 혹시 몰라 정렬
Y_df_with_data = Y_df.loc[set(X_df.index.values)]  # 데이터가 있는 부분만 일단은 추출. test 하는 구간은 어차피 그걸 예측해야하니 생략
Y = Y_df_with_data.values

X_df_scaled = pd.DataFrame(data=X, index=X_df.index, columns=X_df.columns)
values_df = pd.concat([X_df_scaled, Y_df_with_data], axis=1, join='inner')  # 마지막에 Y값을 붙임. colum 명은 0이 됨.
values_df_outer = pd.concat([X_df_scaled, Y_df], axis=1, join='outer')  # 마지막에 Y값을 붙임. colum 명은 0이 됨. 당연하지만 빈곳 투성이일것.

# values = values_df.values


first_layer_node_cnt = int(n_obs * (n_obs - 1)/2)  # 완전 연결 가정한 edge

# split into train and test sets
# values = reframed.values
# n_train_hours = (125 * 24) - n_hours  # 처음 예측해야하는 날짜 계산. test를 해보고 싶지만 일단 해본다. 나중에 for루프로 swell이 발생한 날과 아닌날을 regex이용해서 호출해내는 방식으로 훈련일을 정해야할 것 같다.
# print(values_df.iloc[n_hours + n_train_hours, :].index.name)  # 아마도 n_hour만큼 누락되었을거라 추측. 아무튼 이쪽

# split into input and outputs

# print("train_X.shape, len(train_X), train_y.shape : %s" % train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]

# print("train_X.shape : %s \ntrain_y.shape : %s \ntest_X.shape : %s \ntest_y.shape : %s" % (train_X.shape, train_y.shape, test_X.shape, test_y.shape))
'''
# 빈 accuracy 배열
# accuracy = []
# scorelist = []
mseList = []

scriptName = os.path.basename(os.path.realpath(sys.argv[0]))

num = 0
nan_retain_row = values_df_outer[pd.isnull(values_df_outer).any(axis=1)]
for index, row in nan_retain_row.iterrows():
    print("----------------loop Num : ", num)
    print(index)
    print(row)
    testEnd_time = dt.strptime(index, '%Y-%m-%d %H:%M')  # t
    testStart_time = (testEnd_time - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')  # t-4
    valEnd_time = (testEnd_time - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')  # t-1
    valStart_time = (testEnd_time - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M')  # t-5
    trainEnd_time = (testEnd_time - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
    testEnd_time = (testEnd_time - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')  # 4시간 정도 t-1

    print("EndTrainDate : ", trainEnd_time)
    # print(X_train.shape)
    print("StartValidationDate : ", valStart_time)
    print("EndValidationDate : ", valEnd_time)
    # print(X_val.shape)
    print("StartTestDate : ", testStart_time)
    print("EndTestDate : ", testEnd_time)

    trainX = values_df_outer.loc[:trainEnd_time]
    trainX = scaler.fit_transform(trainX.values)
    valX = values_df_outer.loc[valStart_time:valEnd_time]
    valX = scaler.fit_transform(valX.values)
    testX = values_df_outer.loc[testStart_time:testEnd_time]
    testX = scaler.fit_transform(testX.values)
    # reframed_trainX = series_to_supervised(trainX.values, n_hours, 1)  # t+1 같은 데이터는 별로 필요 없어서 1로 n_out 지정
    # reframed_valX = series_to_supervised(valX, n_hours, 1)  # t+1 같은 데이터는 별로 필요 없어서 1로 n_out 지정
    # reframed.drop(reframed.columns[-n_features:], axis=1, inplace=True)  # 예측해야하는건 이후의 모든 데이터가 아닌 1개의 데이터 뿐.
    X_train, y_train = create_dataset(trainX, n_hours)
    X_val, y_val = create_dataset(valX, n_hours)
    # X_test, y_test = create_dataset(testX, n_hours)
    # reshape training into [samples, timesteps, features]
    train_X = X_train.reshape((X_train.shape[0], n_hours, n_features))
    train_Y = y_train.reshape((y_train.shape[0], 1, n_features))
    val_X = X_val.reshape((X_val.shape[0], n_hours, n_features))
    val_Y = y_val.reshape((y_val.shape[0], 1, n_features))
    test_X = testX.reshape((1, n_hours, n_features))


    # n_batch = gcd(X_train.shape[0], X_val.shape[0])  # 일단 배치사이즈를 대충 결정.
    model = Sequential()
    model.add(LSTM(n_features, batch_input_shape=(1, n_hours, n_features), stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(n_features, batch_input_shape=(1, n_hours, n_features), stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(n_features, batch_input_shape=(1, n_hours, n_features), stateful=True))
    model.add(Dense(n_features))
    model.add(Activation("linear"))
    # model.add(Activation("relu"))
    model.compile(loss='mse', optimizer='adam')

    # 모델 저장 폴더 만들기
    MODEL_DIR = './' + scriptName + ' model_loopNum' + str(num).zfill(2) + '/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    custom_hist = CustomHistory()
    custom_hist.init()
    # fit network # 에포크 2로 해서 14바퀴 돌리니까 93분정도 걸림. 다 돌려면 332분 정도 들텐데 이거에 25배면 138시간 든다. 고작 epoch 50에 약 6일 소요. 시간부족. 단어는 무시하는 걸로 한정시키자. stateless로 한다던가 해야할듯.
    for i in range(epochs):
        model.fit(train_X, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False, validation_data=(val_X, y_val),
                  callbacks=[custom_hist, checkpointer])
        model.reset_states()

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName + ' model1_loopNum' + str(num).zfill(2))
    plt.plot(custom_hist.train_loss)
    plt.plot(custom_hist.val_loss)
    x_len = np.arange(len(custom_hist.val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.show()

    del model
    model = search_best_model(MODEL_DIR)
    evalScore = model.evaluate(X_val, y_val, batch_size=1)
    mseList.append(evalScore)

    # 생각 같아선 추가된 데이터(validation 파트)를 포함해서 좀 더 훈련시켜두고 싶지만 마땅히 validation할 데이터가 없어 과적합 되기 십상. 그냥 이 모델 써서 예상해본다.
    for i in range(epochs):
        model.fit(val_X, y_val, epochs=1, batch_size=1, verbose=0, shuffle=False, validation_data=(train_X, y_train),
                  callbacks=[custom_hist, checkpointer])
        model.reset_states()

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName + ' model2_loopNum' + str(num).zfill(2))
    plt.plot(custom_hist.train_loss)
    plt.plot(custom_hist.val_loss)
    x_len = np.arange(len(custom_hist.val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.show()

    # x_hat = test_X
    # test_start_area_absolute_position = TrainXdf.index.get_loc(int(changeDateToStr(StartTestDate))).start
    prediction = model.predict(test_X, batch_size=1)
    # new_x_hat = np.vstack([np.reshape(X_test[k], (-1, 1)), prediction])
    # x_hat = np.asarray([np.vstack([x_hat[0][1:], np.reshape(new_x_hat, (1, -1))])])
    prediction = scaler.inverse_transform(prediction)
    values_df_outer.loc[index] = prediction

    m, s = divmod((time.time() - start_time), 60)
    print("almost %d minute" % m)
    num += 1
    # break

print("\n about mse: %s" % mseList)
print("mean mse %.7f:" % np.mean(mseList))

values_df_outer.to_csv(scriptName + ' prediction.csv', encoding='utf-8')
