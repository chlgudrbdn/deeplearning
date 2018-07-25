#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

seed = 42
numpy.random.seed(seed)
tf.set_random_seed(seed)
filename = '../dataset/date_And_ironorePrice.csv'

df = pd.read_csv(filename, header=0)
'''
print(df.info())
print(df.head())
'''
dataset = df.values
X = dataset[:,1:]
Y = dataset[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, epochs=200, verbose=1)

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()  # 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해주는 함수
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
# 여기에 RMSE구하면 된다.
