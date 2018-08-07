#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

seed = 42 # 우주의 숫자
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../dataset/full_data_about_iron_ore.csv")  #X는 연도 데이터지만 동시에 종속변수로서 가치가 있다고 간주.

print(df.info())
print(df.head())
dataset = df.values
X = dataset[:,1:]
Y = dataset[:,0]
print("Y")
print(Y)
print("X")
print(X)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
# model = Sequential()
# model.add(Dense(30, input_dim=13, activation='relu'))
# model.add(Dense(keras timeseries 6, activation='relu'))
# model.add(Dense(keras timeseries 6, activation='relu'))
# model.add(Dense(keras timeseries 6, activation='relu'))
# model.add(Dense(keras timeseries 6, activation='relu'))
# model.add(Dense(1))
#
# model.compile(loss='mean_squared_error',
#               optimizer='adam')
#
# model.fit(X_train, Y_train, epochs=200, batch_size=10)
#
# # 예측 값과 실제 값의 비교
# Y_prediction = model.predict(X_test).flatten()  # 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해주는 함수
# for i in range(10):
#     label = Y_test[i]
#     prediction = Y_prediction[i]
#     print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
# # 여기에 RMSE구하면 된다.
