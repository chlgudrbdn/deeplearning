#-*- coding: utf-8 -*-
# https://m.blog.naver.com/silvury/220939233742
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os, sys
from keras.callbacks import ModelCheckpoint,EarlyStopping
import time
start_time = time.time()
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back): # 1이면 그냥 처음부터 끝의 한칸 전까지. 그 이상이면 . range(5)면 0~4 . 1031개 샘플 가진 데이터라면 look_back이 30일때 range가 1000. 즉 0~999=1000번 루프. 1을 빼야할 이유는 모르겠다.
        dataX.append(dataset[i:(i+look_back), 0] )  # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataY.append(dataset[i + look_back, 0]) # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return numpy.array(dataX), numpy.array(dataY) # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦. 이짓을 대충 천번쯤 하는거다.

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
# fix random seed for reproducibility
numpy.random.seed(42)

# load the dataset
filename = os.getcwd() + '\date_And_ironorePrice.csv'
# filename = os.getcwd() + '\dataset\date_And_ironorePrice.csv'
dataframe = pandas.read_csv(filename, usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# hyperparameter tuning section
number_of_var = len(dataframe.columns)
look_back = 25 # 기억력은 1달 일 전후라고 치자. timesteps다.
forecast_ahead = 25
num_epochs = 160
# hyperparameter tuning section
filename = os.path.basename(os.path.realpath(sys.argv[0]))

# 일반적으로 영업일은 250일 쯤 된다. 10-fold validation과 비슷하다.
n_train = dataset.shape[0]-(forecast_ahead*10)  # 총데이터 샘플 수는 2356예상. 35개씩 테스트해서 마지막 개수까지 잘 맞추는 경우를 계산하면 0~1971, 2041,... 2321 식으로 11번 훈련 및 테스팅하는 루프가 돌것(1년 커버하는게 중요).
n_records = dataset.shape[0]  # -(forecast_ahead-1)  # -1은 range가 마지막 수는 포함하지 않기 때문.
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

    # 모델 저장 폴더 만들기
    MODEL_DIR = './'+filename+' model_loopNum'+str(len(average_rmse_list)).zfill(2)+'/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
    # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
    # 학습 자동 중단 설정
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=200)
    train, val, test = dataset[0:i-look_back*2, ], dataset[i-look_back*2: i, ], dataset[i:i+forecast_ahead, ] # 이 경우는 look_back을 사용하는 방식이므로 예측에 충분한 수준의 값을 가져가야한다.
    print('train=%d, val=%d, test=%d' % (len(train), len(val), len(test)))
    trainX, trainY = create_dataset(train, look_back)
    valX, valY = create_dataset(val, look_back)
    # testX, testY = create_dataset(test, look_back)  # forecast_ahead와 look_back이 같으니 이번엔 신경쓸거 없지만 다음 회차엔 신경써야한다.
    print('trainX=%s, trainY=%s' % (trainX.shape, trainY.shape))
    print('valX=%s, valY=%s' % (valX.shape, valY.shape))

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, number_of_var))
    valX = numpy.reshape(valX, (valX.shape[0], look_back, number_of_var))
    # testX = numpy.reshape(testX, (testX.shape[0], look_back, number_of_var))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(number_of_var, look_back, number_of_var), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    custom_hist = CustomHistory()
    custom_hist.init()

    for l in range(num_epochs):
        print("epoch %d" % l)
        model.fit(trainX, trainY, validation_data=(valX, valY), epochs=1, batch_size=1, verbose=0,
                  callbacks=[custom_hist, checkpointer])
        model.reset_states()

    print("--- %s seconds ---" % (time.time() - start_time))
    m, s = divmod((time.time() - start_time), 60)
    print("almost %2f minute" % m)

    # 5. 학습과정 살펴보기
    # plt.plot(custom_hist.train_loss)
    # plt.plot(custom_hist.val_loss)
    # plt.ylim(0.0, 0.15)
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
    file_list.sort()
    model = load_model(MODEL_DIR + file_list[0])

    trainPredict = model.predict(trainX, batch_size=1)
    valPredict = model.predict(valX, batch_size=1)

    xhat = dataset[i-look_back:i, ]  # test셋의 X값 한 세트가 들어간다. 이경우는 값 1개만 예측하면 그만이라지만 좀더 생각해볼 필요가 있다.
    testPredict = numpy.zeros((forecast_ahead, number_of_var))
    for j in range(forecast_ahead):
        prediction = model.predict(numpy.array([xhat]), batch_size=1)
        testPredict[j] = prediction
        xhat = numpy.vstack([xhat[1:], prediction]) # xhat[0]에 있던 녀석은 빼고 재접합해서 xhat[1:]+predction인걸로 한칸 shift해서 예측.

    # invert predictions and answer
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    valPredict = scaler.inverse_transform(valPredict)
    valY = scaler.inverse_transform([valY])
    testPredict = scaler.inverse_transform(testPredict)
    test = scaler.inverse_transform(test)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.4f RMSE' % trainScore)
    valScore = math.sqrt(mean_squared_error(valY[0], valPredict[:, 0]))
    print('Val Score: %.4f RMSE' % valScore)
    testScore = math.sqrt(mean_squared_error(test, testPredict[:,0]))
    print('Test Score: %.4f RMSE' % testScore)

    average_rmse_list.append(testScore)
    if i == (n_records - forecast_ahead): # 루프마지막에.
        plt.figure(figsize=(12, 5))
        plt.plot(numpy.arange(forecast_ahead), testPredict, 'r', label="prediction")
        plt.plot(numpy.arange(forecast_ahead), test[:forecast_ahead], label="test dataset")
        plt.legend()
        plt.show()

        testPredict = numpy.reshape(testPredict, (-1, 5))
        print(testPredict.shape)
        forecast_per_week = testPredict.mean(axis=1)

# print('average loss list:', end=" ")
# print(average_rmse_list)
print('average loss: %.9f' % numpy.mean(average_rmse_list))
forecast_per_week = [round(n, 2) for n in forecast_per_week]
print('forecast_per_week: ', end=" ")
print(forecast_per_week)

print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %2f minute" % m)















# 나중에 모듈로 만들게 이 모델이 다른것 보다 rmse가 작아 우수할 경우 재사용. 위는 그냥 다 주석처리해도 상관없게 코딩해야한다.
# 모델 저장 폴더 만들기
filename = os.path.basename(os.path.realpath(sys.argv[0]))
MODEL_DIR = './' + filename + ' model_loopNum10/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)

train, val = dataset[0:n_records - look_back, ], dataset[n_records - look_back*2: n_records, ]# 이 경우는 look_back을 사용하는 방식이므로 예측에 충분한 수준의 값을 가져가야한다.
print('train=%d, val=%d' % (len(train), len(val)))
trainX, trainY = create_dataset(train, look_back)
valX, valY = create_dataset(val, look_back)
print('trainX=%s, trainY=%s' % (trainX.shape, trainY.shape))
print('valX=%s, valY=%s' % (valX.shape, valY.shape))

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, number_of_var))
valX = numpy.reshape(valX, (valX.shape[0], look_back, number_of_var))
# testX = numpy.reshape(testX, (testX.shape[0], look_back, number_of_var))






# create and fit the LSTM network
model = Sequential()
model.add(LSTM(32, batch_input_shape=(number_of_var, look_back, number_of_var), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

custom_hist = CustomHistory()
custom_hist.init()

for l in range(num_epochs):
    print("epoch %d" % l)
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=1, batch_size=1, verbose=0,
              callbacks=[custom_hist, checkpointer])
    model.reset_states()








print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %2f minute" % m)

# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
file_list.sort()
model = load_model(MODEL_DIR + file_list[0])

trainPredict = model.predict(trainX, batch_size=1)
valPredict = model.predict(valX, batch_size=1)

xhat = dataset[i - look_back:i, ]  # test셋의 X값 한 세트가 들어간다. 이경우는 값 1개만 예측하면 그만이라지만 좀더 생각해볼 필요가 있다.
testPredict = numpy.zeros((forecast_ahead, number_of_var))
for j in range(forecast_ahead):
    prediction = model.predict(numpy.array([xhat]), batch_size=25)
    testPredict[j] = prediction
    xhat = numpy.vstack([xhat[1:], prediction])  # xhat[0]에 있던 녀석은 빼고 재접합해서 xhat[1:]+predction인걸로 한칸 shift해서 예측.

# invert predictions and answer
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
valPredict = scaler.inverse_transform(valPredict)
valY = scaler.inverse_transform([valY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.4f RMSE' % trainScore)
valScore = math.sqrt(mean_squared_error(valY[0], valPredict[:, 0]))
print('Val Score: %.4f RMSE' % valScore)

plt.figure(figsize=(12, 5))
plt.plot(numpy.arange(forecast_ahead), testPredict, 'r', label="prediction")
plt.legend()
plt.show()
testPredict = scaler.inverse_transform(testPredict)
testPredict = numpy.reshape(testPredict, (-1, 5))
print(testPredict.shape)
forecast_per_week = testPredict.mean(axis=1)
print(forecast_per_week)