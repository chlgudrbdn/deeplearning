#-*- coding: utf-8 -*-
# https://m.blog.naver.com/silvury/220939233742
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os, sys
from keras.callbacks import ModelCheckpoint,EarlyStopping
from matplotlib import pyplot
import time
start_time = time.time()
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back): # 1이면 그냥 처음부터 끝의 한칸 전까지. 그 이상이면 . range(5)면 0~4 . 1031개 샘플 가진 데이터라면 look_back이 30일때 range가 1000. 즉 0~999=1000번 루프. 1을 빼야할 이유는 모르겠다.
        dataX.append(dataset[i:(i+look_back), 0] )  # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataY.append(dataset[i + look_back, 0]) # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return numpy.array(dataX), numpy.array(dataY) # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦. 이짓을 대충 천번쯤 하는거다.

# fix random seed for reproducibility
numpy.random.seed(42)

# load the dataset

filename = os.getcwd() + '\date_And_ironorePrice.csv'
# filename = os.getcwd() + '\dataset\date_And_ironorePrice.csv'
# dataframe = pandas.read_csv(r'../dataset/date_And_ironorePrice.csv', usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
dataframe = pandas.read_csv(filename, usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
# dataframe = pandas.read_csv("C:/Users/hyoung-gyu/PycharmProjects/deeplearning/dataset/date_And_ironorePrice.csv", usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
# dataframe = pandas.read_csv('..\dataset\date_And_ironorePrice.csv', usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# X = dataset[:, 1:] # 일단 날짜만 갖고 지지고 볶는것이므로 크게 신경쓸 건 없다.
# Y = dataset[:, 0]


# hyperparameter tuning section
number_of_var = len(dataframe.columns)-1 # 종속변수는 뺀다.
look_back = 30 # 기억력은 30일 전후라고 치자.
timesteps = 5

forecast_ahead = 25
# hyperparameter tuning section
filename=os.path.basename(os.path.realpath(sys.argv[0]))

# Walk Forward Validation로 robustness 체크해 모델의 우수성 비교
# https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
from pandas import Series
# series = Series.from_csv('sunspots.csv', header=0)
# X = series.values
# 일반적으로 영업일은 250일 쯤 된다.
n_train = dataset.shape[0]-(forecast_ahead*10)  # 총데이터 샘플 수는 2356예상. 35개씩 테스트해서 마지막 개수까지 잘 맞추는 경우를 계산하면 0~1971, 2041,... 2321 식으로 11번 훈련 및 테스팅하는 루프가 돌것(1년 커버하는게 중요).
n_records = dataset.shape[0] # -(forecast_ahead-1)  # -1은 range가 마지막 수는 포함하지 않기 때문.
average_rmse_list = []
predictList =[]
print("n_train %d" % n_train)
print("n_records %d" % n_records)
for i in range(n_train, n_records, forecast_ahead):  # 첫 제출일은 적어도 35일 이후 값을 알아야함. 휴일 뺀다면 25일.
    print("loop num : %d" % len(average_rmse_list))
    print("i : %d" % i)

    # 모델 저장 폴더 만들기
    MODEL_DIR = './'+filename+'model_loopNum'+str(len(average_rmse_list)).zfill(2)+'/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
    # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
    # 학습 자동 중단 설정
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    train, test = dataset[0:i, ], dataset[i-look_back : i+forecast_ahead, ] # 이 경우는 look_back을 사용하는 방식이므로 예측에 충분한 수준의 값을 가져가야한다.
    print('train=%d, test=%d' % (len(train), len(test)))
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    print('trainX=%d, trainY=%d' % (len(trainX), len(trainY)))
    print('testX=%d, testY=%d' % (len(testX), len(testY)))

    # trainX, testX = X[0:i, ], X[i:i+35, ]
    # trainY, testY = Y[0:i], Y[i:i + 35]
    # Walk Forward Validation
    # reshape into X=t and Y=t+1

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, testX.shape[1])) # 원본을 따르면 행 개수1571,1,1가 된다. 중간은 time steps 그대로
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1])) # 계산을 위해 형을 바꾸는 식. 773

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(None, look_back)))
    # model.add(LSTM(10, batch_input_shape=(look_back, timesteps, number_of_var), stateful=True))
    # model.add(Dense(5))
    # model.add(Dense(2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)  0 = silent, 1 = progress bar, 2 = one line per epoch.
    # model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)# verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)  0 = silent, 1 = progress bar, 2 = one line per epoch.
    # model.fit(trainX,trainY,nb_epoch=100,validation_split=0.2,verbose=2,callbacks=[early_stopping_callback,checkpointer])
    history = model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=100, batch_size=1, verbose=0, callbacks=[early_stopping_callback, checkpointer])

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    average_rmse_list.append(testScore)
    if i == (n_records - forecast_ahead):
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

# print('average loss list:', end=" ")
# print(average_rmse_list)
print('average loss: %.9f' % numpy.mean(average_rmse_list))

print("--- %s seconds ---" %(time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %2f minute" % m)


##만약 모델이 다른것 보다 rmse가 작아 우수할 경우
# MODEL_DIR = './'+filename+'model_loopNum'+str(len(average_rmse_list)).zfill(2)+'/'
MODEL_DIR = os.getcwd()+'\\'+filename+'model_loopNum'+str(len(average_rmse_list)-1).zfill(2)+'\\'
modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
file_list = os.listdir(MODEL_DIR)  # 루프 가장 마지막 모델 다시 불러오기.
file_list.sort()
print(file_list)
# del model       # 테스트를 위해 메모리 내의 모델을 삭제
model = load_model(MODEL_DIR + file_list[0])

short_memory = []
model.p

# for path, dirs, files in os.walk(MODEL_DIR):
#     print('\nFolder: ', path)
#     if files:
#         for filename in files:
#             print(' Files: ', os.path.join(path, filename))


# # 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
# # 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

train = dataset[:, ]  # 이 경우는 look_back을 사용하는 방식이므로 예측에 충분한 수준의 값을 가져가야한다.
# # print('train=%d, test=%d' % (len(train), len(test)))
trainX, trainY = create_dataset(train, look_back)
# # testX, testY = create_dataset(test, look_back)
print('trainX=%d, trainY=%d' % (len(trainX), len(trainY)))
# # print('testX=%d, testY=%d' % (len(testX), len(testY)))

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, testX.shape[1])) # 원본을 따르면 행 개수1571,1,1가 된다. 중간은 time steps 그대로
# # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1])) # 계산을 위해 형을 바꾸는 식. 773

# # create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None, look_back)))
# model.add(LSTM(10, batch_input_shape=(look_back, timesteps, number_of_var), stateful=True))
# model.add(Dense(5))
# model.add(Dense(2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)  0 = silent, 1 = progress bar, 2 = one line per epoch.
# model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)# verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)  0 = silent, 1 = progress bar, 2 = one line per epoch.
# model.fit(trainX,trainY,nb_epoch=100,validation_split=0.2,verbose=2,callbacks=[early_stopping_callback,checkpointer])
history = model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=100, batch_size=1, verbose=0, callbacks=[early_stopping_callback, checkpointer])
#
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
#
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
#
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
#
# average_rmse_list.append(testScore)
# if i == (n_records- forecast_ahead):
#     pyplot.plot(history.history['loss'], label='train')
#     pyplot.plot(history.history['val_loss'], label='test')
#     pyplot.legend()
#     pyplot.show()




# shift train predictions for plotting

# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict# plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)


# plt.show()


# model = load_model('69-0.0005.hdf5')
# df = pandas.read_csv("../dataset/date_And_ironorePrice-forecast.csv", header=0) #
# datset = df.values
# datset = dataset.astype('float32')
# validationY = datset[:, 0]
# validationY = scaler.fit_transform(validationY )
#
#
# validationY = scaler.inverse_transform(validationY)


# 3. 모델 사용하기
# yhat = model.predict_classes(xhat)