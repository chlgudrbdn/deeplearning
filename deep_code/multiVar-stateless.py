# -*- coding: utf-8 -*-
from datetime import datetime
from math import sqrt
import numpy as np
import os, sys
import pandas
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time
start_time = time.time()
np.random.seed(42)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1] # 딱히 data의 type이 리스트가 아니면 변수 개수 추출. 리스트라면 1.
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1): # 예상에 쓸 t이전 데이터들.
        cols.append(df.shift(i)) # 일단 사용된 코드에선 1이라고 n_in을 정해서 i는 1뿐. 실제론 더 많은 범위의 데이터를 이용할테니 t-1, t-2...가 t시점의 종속변수에 대응.
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)] #
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out): # t +a 앞으로 예상할 범위. 물론 그 예상한 속성들로 종속변수를 계산해야할테니 속성개수만큼 t+a를 예상.
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1) # axis=1은 좌우로 합치는 의미.
    agg.columns = names # 이름 부여. (t-1들)
    # drop rows with NaN values
    if dropnan:# True로만 되긴 한다.
        agg.dropna(inplace=True) # dataframe에서 dropna에 inplace=True이면 NA있는 행은 모두 제거.
    return agg

window_size=25 # look_back과 같다

# load dataset
# 2. 데이터셋 생성하기
filename = os.getcwd() + '\date_And_ironorePrice.csv'
# filename = os.getcwd() + '\dataset\date_And_ironorePrice.csv'
# filename = os.getcwd() + '\\full_data_about_iron_ore.csv'
# filename = os.getcwd() + '\\dataset\\full_data_about_iron_ore.csv'
dataframe = pandas.read_csv(filename, header=0, index_col=1)
values = dataframe.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1) # 각 변수들이 어떻게 시간차나는지를 반영할 용량으로 함.
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# print(reframed.head())  # scaled 된 것들.

# split into train and test sets
# values = reframed.values
values = scaled
n_train_hours = 365 * 24
train = values[:n_train_hours, :]# 1:4 는 1<=x<4 란 의미라 이런식으로 표현
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1] # -1의 바로 앞까지를 X독립변수. -1행을 종속변수로 훈련에 넣는다.
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))  # 3차원 배열로 바꿔두기
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))# testX를 바꿨기 때문에 다시 2차원으로 바꾸는 방식.
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1) # axis=1은 세로로 결합. 근데 어차피 떼어낼 텐데 합칠이유를 잘 모르겠다.
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)