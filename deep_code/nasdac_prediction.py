#-*- coding: utf-8 -*-
# https://m.blog.naver.com/silvury/220939233742
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1): # 1이면 그냥 처음부터 끝의 한칸 전까지. 그 이상이면 . range(5)면 0~4
        a = dataset[i:(i+look_back), 0] # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return numpy.array(dataX), numpy.array(dataY) # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦.

# fix random seed for reproducibility
numpy.random.seed(42)

# load the dataset
filename = '../dataset/date_And_ironorePrice.csv'
# dataframe = pandas.read_csv(r'../dataset/date_And_ironorePrice.csv', usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
# dataframe = pandas.read_csv(filename, usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
dataframe = pandas.read_csv("C:/Users/hyoung-gyu/PycharmProjects/deeplearning/dataset/date_And_ironorePrice.csv", usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
# dataframe = pandas.read_csv('..\dataset\date_And_ironorePrice.csv', usecols=[0]) # 원본은 usecols=[4] 란 옵션 써서 '종가'만 뽑아옴.
dataset = dataframe.values

# delete the commas in the price column (input preprocessing)
# for x in range(len(dataset)):
    # dataset[x][0]=dataset[x][0].replace("," , "")

dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
X = dataset[:, 1:]
Y = dataset[:, 0]
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=42) # 대체할 수 있을지 모르나 일단 차례대로 잘라가도록 둔다.

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1])) # 원본을 따르면 행 개수1571,1,1가 된다. 중간은 time steps 그대로
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1])) # 계산을 위해 형을 바꾸는 식. 773

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)# verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)  0 = silent, 1 = progress bar, 2 = one line per epoch.

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

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()