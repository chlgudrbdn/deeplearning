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

import time
start_time = time.time()

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
cols = collection_df_drop_menu.columns.tolist()
cols = cols[1:] + cols[:1]
collection_df_drop_menu = collection_df_drop_menu[cols]

collection_df_drop_menu_values = collection_df_drop_menu.values
encoder = LabelEncoder()
collection_df_drop_menu_values[:, -1] = encoder.fit_transform(collection_df_drop_menu_values[:, -1])  # 일단 아침, 점심, 저녁을 빼서 임의의 숫자로 만들어둠. 이걸 scaleing할지 고민해봐야한다.

collection_values_float32 = collection_df_drop_menu_values.astype('float32')  # 0번 째 열은 아침점심 저녁 구분인데 이것도 X값으로 쳐야한다. 나중에 귀찮으니 뒤로 옮기자.

test_date_df = pd.read_csv('forecast_date_and_meal_df.csv')
test_date_df = test_date_df.drop(columns=['Unnamed: 0'])
test_date_values = test_date_df.values
encodded_test_date_df = pd.DataFrame(data=encoder.fit_transform(test_date_values[:, 1]), columns=['식사명Encoddeded'])
test_date_df = test_date_df.join(encodded_test_date_df)
test_date_df = test_date_df.set_index(['일자', '식사명'])

test_date_df = pd.merge(test_date_df.reset_index(), collection_df_drop_menu.reset_index(),
                           on=['일자', '식사명'], how='inner').set_index(['일자', '식사명'])
test_date_df = test_date_df.drop(columns=['수량_x', '수량_y'])
cols = test_date_df.columns.tolist()
cols = cols[1:] + cols[:1]
test_date_df = test_date_df[cols]
test_date_df = test_date_df.rename(columns={'식사명Encoddeded': '식사명'})
# test_date_values = test_date_df.values

Y = collection_df_drop_menu_values[:, 0]
X = collection_df_drop_menu_values[:, 1:]

# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)
# X_test = scaler.fit_transform(test_date_df.values.astype('float32'))
X_test = test_date_df.values


number_of_var = X.shape[1]
first_layer_node_cnt = int(((number_of_var-22)*(number_of_var-23))/2)
# first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)
print("first_layer_node_cnt %d" % first_layer_node_cnt)
epochs = 5000
patience_num = 2000
n_fold = 10
kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

rmse_Scores = []
textrmse_Scores = []

trainScoreList = []
valScoreList = []

scriptName = os.path.basename(os.path.realpath(sys.argv[0]))

for train_index, validation_index in kf.split(X):  # 이하 모델을 학습한 뒤 테스트.
    print("loop num : ", len(rmse_Scores)+1)
    # print("TRAIN: %d" % len(train_index), "TEST: %d" % len(validation_index))
    X_train, X_Validation = X[train_index], X[validation_index]
    Y_train, Y_Validation = Y[train_index], Y[validation_index]

    model = Sequential()
    model.add(Dense(first_layer_node_cnt, input_dim=number_of_var-22, activation='relu'))
    edge_num = 2
    while int(first_layer_node_cnt * (edge_num ** (-2))) >= 5 and edge_num < 6:
        model.add(Dense(int(first_layer_node_cnt * (edge_num ** (-2))), activation='relu'))
        model.add(Dropout(0.1))
        edge_num += 1
    model.add(Dense(1))
    print("edge_num : %d" % edge_num)
    model.compile(loss='mse', optimizer='adam')
    # model.compile(loss='mse', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))

    # 모델 저장 폴더 만들기
    MODEL_DIR = './' + scriptName + ' model_loopNum' + str(len(rmse_Scores)).zfill(2) + '/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
    # # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
    # 학습 자동 중단 설정
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    history = model.fit(X_train[:, 22:], Y_train, validation_data=(X_Validation[:, 22:], Y_Validation), epochs=epochs, verbose=0,
                        callbacks=[early_stopping_callback, checkpointer], batch_size=len(X_train))
    # history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, verbose=2, callbacks=[early_stopping_callback, checkpointer])

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName+' model_loopNum'+str(len(rmse_Scores)).zfill(2))
    # 테스트 셋의 오차
    y_loss = history.history['loss']
    y_vloss = history.history['val_loss']
    # 그래프로 표현
    plt.ylim(0.0, 200.0)
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_loss, c="green", label='loss')
    plt.plot(x_len, y_vloss, c="orange", label='val_loss')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.show()

    file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
    file_list = [float(fileName[:-5]) for fileName in file_list]
    file_list.sort()  # 만든날짜 정렬
    model = load_model(MODEL_DIR + '{0:.9f}'.format(file_list[0]) + ".hdf5")
    evalScore = model.evaluate(X_Validation, Y_Validation, batch_size=len(X_Validation))

    rmse_Scores.append(math.sqrt(evalScore))
    prediction_for_val = model.predict(X_Validation[:, 22:], batch_size=len(X_Validation))
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
                                            epochs=50, validation_data=(X_Validation[:, :22], Y_Validation))

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
    textrmse_Scores.append(math.sqrt(evalScore))

    '''
    print("--- %s seconds ---" % (time.time() - start_time))
    m, s = divmod((time.time() - start_time), 60)
    print("almost %2f minute" % m)

print("\n %d fold rmse: %s" % (n_fold, rmse_Scores))
print("mean accuracy %.7f:" % np.mean(rmse_Scores))


"""
    model = Sequential()
    model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu'))
    edge_num = 2
    while int(first_layer_node_cnt * (edge_num**(-2))) >= 5 and edge_num < 6:
        model.add(Dense(int(first_layer_node_cnt * (edge_num**(-2))), activation='relu'))
        model.add(Dropout(0.1))
        edge_num += 1
    model.add(Dense(1))
    print("edge_num : %d" % edge_num)
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    # model.compile(loss='mse', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999), metrics=[rmse])

    # 모델 저장 폴더 만들기
    MODEL_DIR = './'+scriptName+' model_loopNum'+str(len(rmse_Scores)).zfill(2)+'/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR+"{val_rmse:.9f}.hdf5"
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

    print("--- %s seconds ---" % (time.time() - start_time))
    m, s = divmod((time.time() - start_time), 60)
    print("almost %2f minute" % m)

print("\n %d fold rmse: %s" % (n_fold, rmse_Scores))
accuracy = [float(j) for j in rmse_Scores]
print("mean accuracy %.7f:" % np.mean(rmse_Scores))



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
modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
# # 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=2, save_best_only=True)
# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
history = model.fit(X, Y, epochs=epochs, verbose=0, callbacks=[early_stopping_callback], batch_size=len(X))


file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
file_list.sort()
for model_file in file_list:
    print(model_file)
    model = load_model(MODEL_DIR + model_file, custom_objects={'rmse': rmse})

    trainPredict = model.predict(trainX, batch_size=1)
    valPredict = model.predict(valX, batch_size=1)


evalScore = model.evaluate(X, Y, batch_size=len(X))

prediction_for_train = model.predict(X, batch_size=len(X))
prediction_for_test = model.predict(X_test, batch_size=len(X_test))

trainScore = math.sqrt(mean_squared_error(Y_train, prediction_for_train[:, 0]))
print('Train Score: %.4f RMSE' % trainScore)

for multi_index, predic in test_date_df.index.values, prediction_for_test:
    print("%s" % multi_index, ": %d" % predic)
prediction_for_meal_demand = pd.DataFrame(data=prediction_for_test, index=X_test.index.values)
prediction_for_meal_demand.to_csv('prediction_for_test_dnn.csv', encoding='utf-8')
"""
