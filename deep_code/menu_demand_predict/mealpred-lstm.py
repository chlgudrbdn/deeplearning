# -*- coding: utf-8 -*-
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras import backend as K
from keras.models import load_model
import numpy as np
import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rn
import pandas as pd
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import math
from datetime import datetime as dt
from datetime import timedelta


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
        dataX.append(dataset[i:(i + look_back), :-1])  # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataY.append(dataset[i + look_back, -1])  # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return np.array(dataX), np.array(dataY)  # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦. 이짓을 대충 천번쯤 하는거다.


def ordering_meal(mealList):
    mealOrdered = []
    for meal in mealList:
        if meal == '아침식사':
            mealOrdered.append(1)
        elif meal == '점심식사':
            mealOrdered.append(2)
        elif meal == '점심식사2':
            mealOrdered.append(3)
        elif meal == '저녁식사':
            mealOrdered.append(4)
    return mealOrdered


def meal_index_encode(df):
    as_list = df['식사명'].tolist()
    as_list = ordering_meal(as_list)
    df['식사명'] = as_list
    return df


def changeDateToStr(date):
    date = date.strftime('%Y%m%d')
    # date = dt.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    return date


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

# load the dataset
collection_df = pd.read_csv('collection_data_inner.csv', index_col=[0])
collection_df_drop_menu = collection_df.drop(columns=['식사내용'])
collection_df_drop_menu = meal_index_encode(collection_df_drop_menu)
cols = collection_df_drop_menu.columns.tolist()
cols = cols[1:] + cols[:1]
collection_df_drop_menu = collection_df_drop_menu[cols]  # 데이터 프레임에서 식사명 컬럼을 뒤로 미룸
collection_df_drop_menu = collection_df_drop_menu.reset_index().set_index(['일자', '식사명'])
collection_df_drop_menu.sort_index(inplace=True)
collection_df_drop_menu = collection_df_drop_menu.astype('float32')

test_date_df = pd.read_csv('forecast_date_and_meal_df.csv')  # 바로 읽어들이면 아마 날짜-식사명 중복이 존재할 것이다. nan끼리 group by sum하면 0이 되버려서 미리 뺴뒀는데 이렇게 되버림. 여기서 중복제거를 할 수 밖에 없다.
test_date_df = test_date_df.drop(columns=['Unnamed: 0'])
test_date_df = meal_index_encode(test_date_df)  # 이 시점에서 컬럼은 일자, 식사명, 수량 이 있고 식사명을 인코딩.
test_date_df = test_date_df.set_index(['일자', '식사명'])
test_date_df = test_date_df[~test_date_df.index.duplicated(keep='first')]
test_date_df.sort_index(inplace=True)
test_date_df = test_date_df.reset_index()

encodded_test_date_df = pd.DataFrame(data=list(test_date_df['식사명']), columns=['식사명Encoddeded'])
test_date_df = test_date_df.join(encodded_test_date_df)

test_date_df = pd.merge(test_date_df, collection_df_drop_menu.reset_index(),
                        on=['일자', '식사명'], how='inner').set_index(['일자', '식사명'])
test_date_df = test_date_df.drop(columns=['수량_x', '수량_y'])
cols = test_date_df.columns.tolist()
cols = cols[1:] + cols[:1]
test_date_df = test_date_df[cols]
test_date_df = test_date_df.rename(columns={'식사명Encoddeded': '식사명'})
test_date_df = test_date_df.astype('float32')

collection_df_drop_menu = collection_df_drop_menu.reset_index().set_index(['일자'])
cols = collection_df_drop_menu.columns.tolist()
cols = cols[1:] + cols[:1]
collection_df_drop_menu = collection_df_drop_menu[cols]

scaler = MinMaxScaler(feature_range=(0, 1))
cols = collection_df_drop_menu.columns.tolist()

TrainXdf_scaled = scaler.fit_transform(collection_df_drop_menu.values)
TrainXdf = pd.DataFrame(data=TrainXdf_scaled, index=collection_df_drop_menu.index, columns=collection_df_drop_menu.columns)  # 최종적으로 사용.

TestXdf_scaled = scaler.fit_transform(test_date_df.values)
TestXdf = pd.DataFrame(data=TestXdf_scaled, index=test_date_df.index, columns=test_date_df.columns)  # 최종적으로 사용.

scriptName = os.path.basename(os.path.realpath(sys.argv[0]))

rmse_Scores = []
trainScoreList = []
valScoreList = []

# hyper param
number_of_var = len(cols) - 1
first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)
print("first_layer_node_cnt %d" % first_layer_node_cnt)
epochs = 50
patience_num = 10
look_back = 4 * 8  # test date 날짜 차이가 최소 8일 정도 되는 것 같다. 공백과 공백 사이 최소 4일(점심2가 없는 날은 다행이 없으므로 16)
forecast_ahead = 4 * 3  # 애초에 3일 뒤 것을 맞추는 문제다.
###################

StartTrainDate = dt.strptime(str(20090803), '%Y%m%d').date() + timedelta(days=1)  # 20090803은 데이터 결락이 없는 마지막 날. 2010년 부터 추측해 제출하면 되므로 이게 더 좋을 것이다.
test_only_date = test_date_df.index.levels[0].tolist()
for num in range(0, len(test_only_date), 3):  # 50회 루프가 있을 것이다.
    EndTrainDate = dt.strptime(str(test_only_date[num]), '%Y%m%d').date() - timedelta(days=5)  # 2014-06-04 - 5 = 2010-05-30까지
    StartValidationDate = EndTrainDate + timedelta(days=1)  # 20100710
    EndValidationDate = StartValidationDate + timedelta(days=2)  # 20100712
    StartTestDate = EndValidationDate + timedelta(days=1)  # 20100713
    EndTestDate = StartTestDate + timedelta(days=2)  # 20100715

    print("StartTrainDate : ", StartTrainDate)  # 20100713
    print("EndTrainDate : ", EndTrainDate)
    print("StartValidationDate : ", StartValidationDate)
    print("EndValidationDate : ", EndValidationDate)
    print("StartTestDate : ", StartTestDate)
    print("EndTestDate : ", EndTestDate)

    X_train = TrainXdf.loc[changeDateToStr(StartTrainDate):changeDateToStr(EndTrainDate)].values
    # Y_train = X_train[:, -1]
    X_train, y_train = create_dataset(X_train, look_back)

    X_val = TrainXdf.loc[changeDateToStr(StartValidationDate):changeDateToStr(EndValidationDate)].values
    Y_val = X_val[:, -1]

    X_test_for_train = np.vstack([X_train, X_val])  # 제출용 날짜 예측을 위한 훈련 데이터 재구성.
    Y_test_for_train = np.vstack([Y_train, Y_val])  # 제출용 날짜 예측을 위한 훈련 데이터 재구성.


    X_test = TestXdf.loc[changeDateToStr(StartTestDate):changeDateToStr(EndTestDate)]  # 별도로 안한다. 어차피 0~100사이 정규화 되어 있기도 하고.
    Y_test = []








    train_X = train_X.reshape((train_X.shape[0], look_back, train_X.shape[2]))
    test_X = test_X.reshape((test_X.shape[0], look_back, train_X.shape[2]))


    if num != len(test_only_date) - 3: # 마지막 루프만 아니면
        StartTrainDate = EndTestDate + timedelta(days=1)  # 다음 루프때 쓸 train data 구간 규정
    m, s = divmod((time.time() - start_time), 60)
    print("almost %d minute" % m)

'''
for train_index, validation_index in kf.split(X):  # 이하 모델을 학습한 뒤 테스트.
    print("loop num : ", len(rmse_Scores)+1)
    # print("TRAIN: %d" % len(train_index), "TEST: %d" % len(validation_index))
    X_train, X_Validation = X[train_index], X[validation_index]
    Y_train, Y_Validation = Y[train_index], Y[validation_index]

    model = Sequential()
    model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu'))
    edge_num = 2
    while int(first_layer_node_cnt * (edge_num ** (-2))) >= 5 and edge_num < 6:
        model.add(Dense(int(first_layer_node_cnt * (edge_num ** (-2))), activation='relu'))
        model.add(Dropout(0.1))
        edge_num += 1
    model.add(Dense(1))
    print("edge_num : %d" % edge_num)
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    # model.compile(loss='mse', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999), metrics=[rmse])

    # 모델 저장 폴더 만들기
    MODEL_DIR = './' + scriptName + ' model_loopNum' + str(len(rmse_Scores)).zfill(2) + '/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR + "{val_rmse:.9f}.hdf5"
    # # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_rmse', verbose=2, save_best_only=True)
    # 학습 자동 중단 설정
    early_stopping_callback = EarlyStopping(monitor='val_rmse', patience=patience_num)
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    history = model.fit(X_train, Y_train, validation_data=(X_Validation, Y_Validation), epochs=epochs, verbose=0,
                        callbacks=[early_stopping_callback], batch_size=len(X_train))
    # history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, verbose=2, callbacks=[early_stopping_callback, checkpointer])

    plt.figure(figsize=(8, 8))
    # 테스트 셋의 오차
    y_rmse = history.history['rmse']
    y_vrmse = history.history['val_rmse']
    y_loss = history.history['loss']
    y_vloss = history.history['val_loss']
    # 그래프로 표현
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_rmse, c="blue", label='y_rmse')
    plt.plot(x_len, y_vrmse, c="red", label='y_vrmse')
    plt.plot(x_len, y_loss, c="green", label='loss')
    plt.plot(x_len, y_vloss, c="orange", label='val_loss')

    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.show()

    evalScore = model.evaluate(X_Validation, Y_Validation, batch_size=len(X_Validation))

    prediction_for_train = model.predict(X_train, batch_size=len(X_Validation))
    prediction_for_val = model.predict(X_Validation, batch_size=len(X_Validation))

    trainScore = math.sqrt(mean_squared_error(Y_train, prediction_for_train[:, 0]))
    print('Train Score: %.4f RMSE' % trainScore)
    trainScoreList.append(trainScore)
    valScore = math.sqrt(mean_squared_error(X_Validation, prediction_for_val[:, 0]))
    print('Val Score: %.4f RMSE' % valScore)

    # print("predict : %s" % prediction_for_val)
    # print("real    : %s" % Y_Validation)
    rmse_Scores.append(evalScore[1])
    '''

'''
    text_anal_model = Sequential()
    text_anal_model.add(Embedding(3076, 22))  # Embedding층은 데이터 전처리 과정 통해 입력된 값을 받아 다음 층이 알아들을 수 있는 형태로 변환하는 역할. (불러온 단어의 총 개수, 기사당 단어 수). 1000가지 단어를 각 샘플마다 100개씩 feature로 갖고 있다.
    text_anal_model.add(LSTM(11*21, activation='relu'))
    # text_anal_model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))  # MNIST_Deep 에선 2차원 행렬 합성곱을 했지만 이경우는 1차원.
    # text_anal_model.add(MaxPooling1D(pool_size=4))
    # padding: 바깥에 0을 채워넣냐 마냐.. "valid" 는 패딩 없단 소리. "same" 인풋과 같은 길이의 패딩 0 붙임(길이 조절은 불가). 결과적으로 출력 이미지 사이즈가 입력과 동일. "causal" 확대한 합성곱의 결과. 모델이 시간 순서를 위반해서는 안되는 시간 데이터를 모델링 할 때 유용.
    # strides는 다음칸을 움직이는 칸수 정도로 보면 된다. 2이고 왼쪽에서 오른쪽 2칸 움직이고 다음 행으로 갈 땐 2칸 아래로 가는 식.
    text_anal_model.add(Dense(1, activation='relu'))  # RNN은 여러 상황에서 쓰일수 있는데 다수 입력 단일 출력, 단일 입력 다수 출력도 가능(사진의 여러 요소를 추출해 캡션 만들 때 사용). 이건 후자. 다수입력 다수 출력도 가능.
    text_anal_model.compile(loss='mse', optimizer='adam')

    # 모델 저장 폴더 만들기
    MODEL_DIR = './'+scriptName+' model_loopNum'+str(len(rmse_Scores)).zfill(2)+'/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
    # # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
    # 학습 자동 중단 설정
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)

    text_anal_history = text_anal_model.fit(X_train[:, :22], Y_train, batch_size=len(X_train), verbose=0,
                                            epochs=epochs, validation_data=(X_Validation[:, :22], Y_Validation))

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName+' model_loopNum'+str(len(rmse_Scores)).zfill(2) )
    y_loss = text_anal_history.history['loss']
    y_vloss = text_anal_history.history['val_loss']
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_loss, c="green", label='loss')
    plt.plot(x_len, y_vloss, c="orange", label='val_loss')

    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    evalScore = text_anal_model.evaluate(X_Validation, Y_Validation, batch_size=len(X_Validation))
    rmse_Scores.append(evalScore)

    '''

# print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %2f minute" % m)

print("\nrmse: %s" % rmse_Scores)
print("mean rmse %.7f:" % np.mean(rmse_Scores))






