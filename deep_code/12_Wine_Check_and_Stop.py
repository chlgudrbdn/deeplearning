from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import os, sys
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
import time
start_time = time.time()
# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('../dataset/wine.csv', header=None)
df = df_pre.sample(frac=0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

# 모델 저장 폴더 만들기
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

modelpath="./model/{val_loss:.4f}.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

model.fit(X_train, Y_train, validation_split=0.2, epochs=500, batch_size=500, verbose=2, validation_data=(X_test, Y_test),
          callbacks=[early_stopping_callback, checkpointer])


testScore = model.evaluate(X_test, Y_test, verbose=0)
# testScore = math.sqrt(testScore)
print('Test Score: %s RMSE' % testScore)

predictY = model.predict(X_test)
testScore = math.sqrt(mean_squared_error(Y_test, predictY[:, 0]))
print('Test Score: %.9f RMSE' % testScore)


MODEL_DIR = os.getcwd()+'\\model\\'
file_list = os.listdir(MODEL_DIR)  # 루프 가장 마지막 모델 다시 불러오기.
file_list.sort()
# print(file_list)
del model       # 테스트를 위해 메모리 내의 모델을 삭제
print(file_list[0])
model = load_model(MODEL_DIR + file_list[0])

testScore = model.evaluate(X_test, Y_test, verbose=0)
# testScore = math.sqrt(testScore)
print('Test Score: %s RMSE' % testScore)

fore_predict = model.predict(X_test)
testScore = math.sqrt(mean_squared_error(Y_test, fore_predict[:, 0]))
print('Test Score: %.9f RMSE' % testScore)
print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
