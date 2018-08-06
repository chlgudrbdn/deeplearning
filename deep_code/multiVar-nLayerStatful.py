# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import tensorflow as tf
import keras
from keras import backend as K
from keras.backend import manual_variable_initialization

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, sys
from keras.callbacks import ModelCheckpoint, EarlyStopping
import random as rn
import time

start_time = time.time()

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 데이터셋 생성 함수
def create_dataset(data, window_size):
    # dataset_X, dataset_Y = [], []  # data가 50개라 치자.
    # for i in range(len(data) - window_size):  # range(50-25). 0~25 = 26회 루프.
    #     subset = data[i:(i + window_size + 1)]  # [0:26]=0~25=26개. [25:51]=25~50=26
    #     # print("i : %d" % i)
    #     # print("len(subset) : %d" % len(subset))
    #     for si in range(len(subset) - 1):  # range(26-1=25)=0~24=25회. 루프.
    #         dataset_X.append(subset[si])  # 25개
    #     dataset_Y.append(subset[window_size])  # Y는 여기에. 26번째. 번호는 25니까. 종속변수는 특정 시계열의 모든 변수.
    # return np.array(dataset_X), np.array(dataset_Y)
    dataX, dataY = [], [] # 총 50에 윈도우 크기 25라 치면. 0~24=25회 0~24와 25를 대응.
    for i in range(len(dataset)-window_size): # 1이면 그냥 처음부터 끝의 한칸 전까지. 그 이상이면 . range(5)면 0~4
        dataX.append(dataset[i:(i+window_size), ])  # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataY.append(dataset[i + window_size, ])  # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return np.array(dataX), np.array(dataY) # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦.

# fix random seed for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(42)
rn.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# manual_variable_initialization(True)
tf.global_variables_initializer()

# 1. 데이터 준비하기
# 하이퍼 파라미터 정의
window_size = 25 # look_back과 같다
forecast_ahead = 35
filename = os.path.basename(os.path.realpath(sys.argv[0]))
num_epochs = 300
pred_count = 35  # 최대 예측 개수 정의

# 2. 데이터셋 생성하기
filename = os.getcwd() + '\date_And_ironorePrice.csv'
# filename = os.getcwd() + '\dataset\date_And_ironorePrice.csv'
# filename = os.getcwd() + '\\full_data_about_iron_ore.csv'
# filename = os.getcwd() + '\\dataset\\full_data_about_iron_ore.csv'
dataframe = pandas.read_csv(filename)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

x_train, y_train = create_dataset(dataset, window_size=window_size)

var_num = dataset.shape[1]
first_layer_node_cnt = int(var_num*(var_num-1)/2)

# 입력을 (샘플 수, 타임스텝, 특성 수)로 형태 변환
x_train = np.reshape(x_train, (x_train.shape[0]-window_size, window_size, var_num))
y_train = np.reshape(y_train, (y_train.shape[0], 1, var_num))


# 3. 모델 구성하기
model = Sequential()
model.add(LSTM(first_layer_node_cnt, batch_input_shape=(var_num, window_size, var_num), stateful=True, return_sequences =True)) # 15
model.add(Dense(int(first_layer_node_cnt/2), activation='relu')) # 7
# model.add(Dropout(0.3))
model.add(Dense(int(first_layer_node_cnt/4), activation='relu')) # 3
# model.add(Dropout(0.3))
# model.add(Dense(int(first_layer_node_cnt/8), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(int(first_layer_node_cnt/16), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(int(first_layer_node_cnt/32), activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(var_num, activation='relu'))

# 4. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 5. 모델 학습시키기
history = LossHistory()  # 손실 이력 객체 생성
history.init()

script_name = os.path.basename(os.path.realpath(sys.argv[0]))
average_rmse_list = []

# 모델 저장 폴더 만들기
MODEL_DIR = './'+script_name+'model_loopNum'+str(len(average_rmse_list)).zfill(2)+'/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
# # 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='loss', verbose=2, save_best_only=True)
# # 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False,
              callbacks=[history, early_stopping_callback, checkpointer])  # 50 is X.shape[0]
    model.reset_states()

# 6. 학습과정 살펴보기
pyplot.plot(history.losses)
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train'], loc='upper left')
pyplot.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train, batch_size=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
model.reset_states()

# 8. 모델 사용하기

# 곡 전체 예측
# seq_in = x_train[dataset.shape[0]-(window_size*10): dataset.shape[0]-(window_size*10)+window_size, ]
# seq_out = seq_in
# seq_in_featrues = []
# # for si in seq_in:
# #     features = code2features(si)
# #     seq_in_featrues.append(features)
#
# for i in range(pred_count):
#     sample_in = np.array(seq_in_featrues)
#     sample_in = np.reshape(sample_in, (window_size, 4, 2))  # 샘플 수, 타입스텝 수, 속성 수
#     pred_out = model.predict(sample_in)
#     idx = np.argmax(pred_out)
#     seq_out.append(idx2code[idx])
#
#     seq_in_featrues.append(features)
#     seq_in_featrues.pop(0)
#     print("seq_in_featrues")
#     print(seq_in_featrues)
#
# model.reset_states()
#
# print("full song prediction : ", seq_out)
#
#
#
#
# # Walk Forward Validation로 robustness 체크해 모델의 우수성 비교. 이하 출처에서 개념 추출.
# # https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
# # 일반적으로 영업일은 250일 쯤 된다.
# n_train = dataset.shape[0]-(window_size*10)  # 총데이터 샘플 수는 2356예상. 35개씩 테스트해서 마지막 개수까지 잘 맞추는 경우를 계산하면 0~1971, 2041,... 2321 식으로 11번 훈련 및 테스팅하는 루프가 돌것(1년 커버하는게 중요).
# n_records = dataset.shape[0] # -(forecast_ahead-1)  # -1은 range가 마지막 수는 포함하지 않기 때문.
# average_rmse_list = []
# predictList =[]
# print("n_train %d" % n_train)
# print("n_records %d" % n_records)
# for i in range(n_train, n_records, forecast_ahead):  # 첫 제출일은 적어도 35일 이후 값을 알아야함. 휴일 뺀다면 25일.
#     print("loop num : %d" % len(average_rmse_list))
#     print("i : %d" % i)
#
#     # 모델 저장 폴더 만들기
#     MODEL_DIR = './'+filename+'model_loopNum'+str(len(average_rmse_list)).zfill(2)+'/'
#     if not os.path.exists(MODEL_DIR):
#         os.mkdir(MODEL_DIR)
#     modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
#     # 모델 업데이트 및 저장
#     checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
#     # 학습 자동 중단 설정
#     early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
#
#     # 곡 전체 예측
#     seq_in = ['g8', 'e8', 'e4', 'f8']
#     seq_in_featrues = []
#     # for si in seq_in:
#     #     features = code2features(si)
#     #     seq_in_featrues.append(features)
#
#     for i in range(pred_count):
#         sample_in = np.array(seq_in_featrues)
#         sample_in = np.reshape(sample_in, (1, 4, 2))  # 샘플 수, 타입스텝 수, 속성 수
#         pred_out = model.predict(sample_in)
#         idx = np.argmax(pred_out)
#         seq_out.append(idx2code[idx])
#
#         features = code2features(idx2code[idx])
#         seq_in_featrues.append(features)
#         seq_in_featrues.pop(0)
#
#     model.reset_states()
#
#     print("full song prediction : ", seq_out)
#
#
#
#
#
#
#     train, test = dataset[0:i, ], dataset[i-window_size : i+forecast_ahead, ] # 이 경우는 look_back을 사용하는 방식이므로 예측에 충분한 수준의 값을 가져가야한다.
#     print('train=%d, test=%d' % (len(train), len(test)))
#     trainX, trainY = create_dataset(train, window_size)
#     testX, testY = create_dataset(test, window_size)
#     print('trainX=%d, trainY=%d' % (len(trainX), len(trainY)))
#     print('testX=%d, testY=%d' % (len(testX), len(testY)))
#
#     # reshape input to be [samples, time steps, features]
#     trainX = np.reshape(trainX, (trainX.shape[0], 1, testX.shape[1])) # 원본을 따르면 행 개수1571,1,1가 된다. 중간은 time steps 그대로
#     testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1])) # 계산을 위해 형을 바꾸는 식. 773
#
#     # create and fit the LSTM network
#     model = Sequential()
#     model.add(LSTM(4, input_shape=(None, window_size)))
#     # model.add(LSTM(10, batch_input_shape=(look_back, timesteps, number_of_var), stateful=True))
#     # model.add(Dense(5))
#     # model.add(Dense(2))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     # verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)  0 = silent, 1 = progress bar, 2 = one line per epoch.
#     # model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)# verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)  0 = silent, 1 = progress bar, 2 = one line per epoch.
#     # model.fit(trainX,trainY,nb_epoch=100,validation_split=0.2,verbose=2,callbacks=[early_stopping_callback,checkpointer])
#     history = model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=100, batch_size=1, verbose=0, callbacks=[early_stopping_callback, checkpointer])
#
#     # make predictions
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)
#
#     # invert predictions
#     trainPredict = scaler.inverse_transform(trainPredict)
#     trainY = scaler.inverse_transform([trainY])
#     testPredict = scaler.inverse_transform(testPredict)
#     testY = scaler.inverse_transform([testY])
#
#     # calculate root mean squared error
#     trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#     print('Train Score: %.2f RMSE' % (trainScore))
#     testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#     print('Test Score: %.2f RMSE' % (testScore))
#
#     average_rmse_list.append(testScore)
#     if i == (n_records - forecast_ahead):
#         pyplot.plot(history.history['loss'], label='train')
#         pyplot.plot(history.history['val_loss'], label='test')
#         pyplot.legend()
#         pyplot.show()
#
#
#



print("--- %s seconds ---" %(time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %2f minute" % m)
