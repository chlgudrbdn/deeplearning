#-*- coding: utf-8 -*-
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import pandas
import numpy
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os, sys
from keras.callbacks import ModelCheckpoint,EarlyStopping
seed = 42
numpy.random.seed(seed)
tf.set_random_seed(seed)
filename = os.getcwd() + '\date_And_ironorePrice.csv'
# filename = os.getcwd() + '\dataset\date_And_ironorePrice.csv'
dataframe = pandas.read_csv(filename)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
'''
print(df.info())
print(df.head())
'''
X = dataset[:, 1:]
y = dataset[:, 0]
number_of_var = X.shape[1] # 4 기대.
# first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)  # nC2 6


forecast_ahead = 25


n_train = y.shape[0] - (forecast_ahead*10)
n_records = y.shape[0]
average_rmse_list = []
predictList = []
forecast_per_week = []
print("n_train %d" % n_train)
print("n_records %d" % n_records)
# Walk Forward Validation로 robustness 체크해 모델의 우수성 비교. 개념은 아래 출처에서.
# https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
for i in range(n_train, n_records, forecast_ahead):  # 첫 제출일은 적어도 35일 이후 값을 알아야함. 휴일 뺀다면 25일.
    print("loop num : %d" % len(average_rmse_list))
    print("i : %d" % i)
    X_train, X_test = X[0:i, ], X[i: i+forecast_ahead, ]
    y_train, y_test = y[0:i, ], y[i: i+forecast_ahead, ]
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # 모델 저장 폴더 만들기
    MODEL_DIR = './' + filename + ' model_loopNum' + str(len(average_rmse_list)).zfill(2) + '/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
    # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30)

    model = Sequential()
    model.add(Dense(32, input_dim=number_of_var, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))  # 3
    # model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(4, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(2, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(int(first_layer_node_cnt/32), activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(1), activation='relu')
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=1, verbose=0, callbacks=[early_stopping_callback, checkpointer], validation_data=(X_test, y_test))

    # 예측 값과 실제 값의 비교
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.9f RMSE' % testScore)
    y_prediction = model.predict(X_test).flatten()  # 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해주는 함수
    y_prediction = scaler.inverse_transform(y_prediction)

    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(y_test, y_prediction[:, 0]))
    print('Test Score: %.9f RMSE' % testScore)

    average_rmse_list.append(testScore)


filename = os.getcwd() + '\date_And_ironorePrice-forecast.csv'
# filename = os.getcwd() + '\dataset\date_And_ironorePrice-forecast.csv'
df = pandas.read_csv(filename)
val_dat = df.values
val_dat = val_dat.astype('float32')
val_dat = scaler.fit_transform(val_dat)

# 만약 이 모델이 다른것 보다 rmse가 작아 우수할 경우 재사용. 위는 그냥 다 주석처리해도 상관없다.
MODEL_DIR = os.getcwd()+'\\'+filename+'model_loopNum'+str(9).zfill(2)+'\\'
modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
file_list = os.listdir(MODEL_DIR)  # 루프 가장 마지막 모델 다시 불러오기.
file_list.sort()
print(file_list)
del model       # 테스트를 위해 메모리 내의 모델을 삭제
model = load_model(MODEL_DIR + file_list[0])
xhat = dataset[-25:, ]
fore_predict = numpy.zeros((forecast_ahead, number_of_var))
# for k in range(forecast_ahead):
prediction = model.predict(numpy.array([xhat]), batch_size=1)
# fore_predict[k] = prediction
# xhat = numpy.vstack([xhat[1:], prediction])

fore_predict = numpy.reshape(fore_predict, (-1, 5))
forecast_per_week = fore_predict.mean(axis=1)
print(forecast_per_week)
