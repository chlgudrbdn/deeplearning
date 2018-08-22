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
from keras.models import load_model
from keras import backend as K
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


def ordering_meal(mealList):
    mealOrdered = []
    for meal in mealList:
        if meal == '아침식사':
            mealOrdered.append(1)
        elif meal == '점심식사':
            mealOrdered.append(2)
        elif meal == '점심식사2':
            mealOrdered.append(3)
        elif meal == '저녁식사':
            mealOrdered.append(4)
    return mealOrdered


def meal_index_encode(df):
    as_list = df['식사명'].tolist()
    as_list = ordering_meal(as_list)
    df['식사명'] = as_list
    return df


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
collection_df_drop_menu = meal_index_encode(collection_df_drop_menu)
cols = collection_df_drop_menu.columns.tolist()
cols = cols[1:] + cols[:1]
collection_df_drop_menu = collection_df_drop_menu[cols]
collection_df_drop_menu = collection_df_drop_menu.reset_index().set_index(['일자', '식사명'])
collection_df_drop_menu.sort_index(inplace=True)
collection_df_drop_menu = collection_df_drop_menu.astype('float32')

test_date_df = pd.read_csv('forecast_date_and_meal_df.csv')
test_date_df = test_date_df.drop(columns=['Unnamed: 0'])
test_date_df = meal_index_encode(test_date_df)  # 이 시점에서 컬럼은 일자, 식사명, 수량 이 있고 식사명을 인코딩.
test_date_df = test_date_df.set_index(['일자', '식사명'])
test_date_df = test_date_df[~test_date_df.index.duplicated(keep='first')]  # 중복을 빼는데 이상하게 596까지 줄어듬.
test_date_df.sort_index(inplace=True)
test_date_df = test_date_df.reset_index()

encodded_test_date_df = pd.DataFrame(data=list(test_date_df['식사명']), columns=['식사명Encoddeded'])
test_date_df = test_date_df.join(encodded_test_date_df)

test_date_df = pd.merge(test_date_df.reset_index(), collection_df_drop_menu.reset_index(),
                           on=['일자', '식사명'], how='inner').set_index(['일자', '식사명'])
test_date_df = test_date_df.drop(columns=['수량_x', '수량_y','index'])
cols = test_date_df.columns.tolist()
cols = cols[1:] + cols[:1]
test_date_df = test_date_df[cols]
test_date_df = test_date_df.rename(columns={'식사명Encoddeded': '식사명'})
test_date_df = test_date_df.astype('float32')

collection_df_drop_menu = collection_df_drop_menu.reset_index().set_index(['일자'])
cols = collection_df_drop_menu.columns.tolist()
cols = cols[1:] + cols[:1]
collection_df_drop_menu = collection_df_drop_menu[cols]

scaler = MinMaxScaler(feature_range=(0, 1))
cols = collection_df_drop_menu.columns.tolist()

TrainXdf_scaled = scaler.fit_transform(collection_df_drop_menu.values[:, 1:])
TrainXdf = pd.DataFrame(data=TrainXdf_scaled, index=collection_df_drop_menu.index, columns=collection_df_drop_menu.columns[1:])  # 최종적으로 사용.
TrainYdf = pd.DataFrame(data=collection_df_drop_menu.values[:, 0], index=collection_df_drop_menu.index, columns=[collection_df_drop_menu.columns[0]])

TestXdf_scaled = scaler.fit_transform(test_date_df.values)
TestXdf = pd.DataFrame(data=TestXdf_scaled, index=test_date_df.index, columns=test_date_df.columns)  # 최종적으로 사용.

X = TrainXdf.values
Y = TrainYdf.values
X_test = TestXdf.values

number_of_var = len(cols) - 1  # 1빼는건 종속편수가 포함되서
first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)
print("first_layer_node_cnt %d" % first_layer_node_cnt)
epochs = 300
patience_num = 200
n_fold = 10
kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

rmse_Scores = []
trainScoreList = []
valScoreList = []

scriptName = os.path.basename(os.path.realpath(sys.argv[0]))


model = Sequential()
model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu', kernel_initializer='random_normal'))
edge_num = 2
while int(first_layer_node_cnt * (edge_num ** (-2))) >= 5 and edge_num < 6:
    model.add(Dense(int(first_layer_node_cnt * (edge_num ** (-2))), activation='relu', kernel_initializer='random_normal'))
    model.add(Dropout(0.3))
    edge_num += 1
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
# model.compile(loss='mse', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999)])

# 모델 저장 폴더 만들기
MODEL_DIR = './' + scriptName + ' model_loopNum' + str(len(rmse_Scores)).zfill(2) + '/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
# # 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
history = model.fit(X, Y, epochs=epochs, validation_split=0.1, verbose=0,
                    callbacks=[checkpointer, early_stopping_callback], batch_size=32)

plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName + ' model_loopNum' + str(len(rmse_Scores)).zfill(2))
# 테스트 셋의 오차
y_loss = history.history['loss']
y_vloss = history.history['val_loss']
# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, c="green", label='loss')
plt.plot(x_len, y_vloss, c="orange", label='val_loss')
plt.legend(loc='upper left')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
file_list = [float(fileName[:-5]) for fileName in file_list]
file_list.sort()  # 만든날짜 정렬
print(str(file_list[0]) + ".hdf5")
model = load_model(MODEL_DIR + '{0:.9f}'.format(file_list[0]) + ".hdf5")

evalScore = model.evaluate(X, Y, batch_size=len(X))  # val_loss가 아닌 loss 값이나올 뿐이다.
print("loss evalScore", end=" ")
print(evalScore)  # print(evalScore)로 확인 결과 loss, rmse. 앞에걸 sqrt하면 그냥 일일이 계산한 결과와 같은게 튀어나오는 것 같다.

prediction_for_train = model.predict(X, batch_size=len(X))
trainScore = math.sqrt(mean_squared_error(Y, prediction_for_train[:, 0]))
print('Train Score: %.9f RMSE' % trainScore)

prediction_for_test = model.predict(X_test, batch_size=len(X_test))
# for i in range(len(prediction_for_test)):
#     print("date: %s" % test_date_df.index.values[i][0], "meal: %s" % test_date_df.index.values[i][1],
#           "prediction: %.4f:" % prediction_for_test[i])

onlyDateInTest_df = list(set(list(test_date_df.index.levels[0])))
onlyDateInTest_df.sort()
onlyDateInTest_df = [onlyDateInTest_df[date_idx] for date_idx in range(2, len(onlyDateInTest_df), 3)]
prediction_for_meal_demand = pd.DataFrame(data=prediction_for_test, index=test_date_df.index)
prediction_for_meal_demand = prediction_for_meal_demand.loc[onlyDateInTest_df]

prediction_for_meal_demand = prediction_for_meal_demand.values.reshape((-1, 4))

prediction_for_meal_demand_df = pd.DataFrame(data=prediction_for_meal_demand, index=onlyDateInTest_df, columns=['아침식사', '점심식사', '점심식사2', '저녁식사'])
prediction_for_meal_demand_df.to_csv('prediction_for_test_dnn.csv', encoding='utf-8')

print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %d minute" % m)
