# -*- coding: utf-8 -*-

import numpy as np
import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rn
import pandas as pd
import keras
from sklearn.metrics import mean_squared_error
import math
from datetime import datetime as dt
from datetime import timedelta
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras import backend as K
from keras.models import load_model

import time
start_time = time.time()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):  # 1이면 그냥 처음부터 끝의 한칸 전까지. 그 이상이면 . range(5)면 0~4 . 1031개 샘플 가진 데이터라면 look_back이 30일때 range가 1000. 즉 0~999=1000번 루프. 1을 빼야할 이유는 모르겠다.
        dataX.append(dataset[i:(i + look_back), ])  # 1이면 2개씩 dataX에 추가. i가 0이면 0~1까지.
        dataY.append(dataset[i + look_back, -1])  # i 가 0이면 1 하나만. X와 비교하면 2대 1 대응이 되는셈.
    return np.array(dataX), np.array(dataY)  # 즉 look_back은 1대 look_back+1만큼 Y와 X를 대응 시켜 예측하게 만듦. 이짓을 대충 천번쯤 하는거다.


class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def search_best_model(MODEL_DIR):
    file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
    file_list = [float(fileName[:-5]) for fileName in file_list]
    file_list.sort()  # 만든날짜 정렬
    model = load_model(MODEL_DIR + '{0:.9f}'.format(file_list[0]) + ".hdf5")
    return model


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


def changeDateToStr(date):
    date = date.strftime('%Y%m%d')
    # date = dt.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    return date


def gcd(a, b):
    if a < b:
        (a, b) = (b, a)
    while b != 0:
        (a, b) = (b, a % b)
    return a
# 출처: http://codepractice.tistory.com/65 [코딩 연습]


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
collection_df_drop_menu = collection_df_drop_menu[cols]  # 데이터 프레임에서 식사명 컬럼을 뒤로 미룸
collection_df_drop_menu = collection_df_drop_menu.reset_index()
encodded_mealDemand_df = pd.DataFrame(data=list(collection_df_drop_menu['식사명']), columns=['식사명Encoddeded'])
collection_df_drop_menu = collection_df_drop_menu.join(encodded_mealDemand_df)
collection_df_drop_menu = collection_df_drop_menu.set_index(['일자', '식사명'])
# collection_df_drop_menu = collection_df_drop_menu.rename(columns={'식사명Encoddeded': '식사명'})
collection_df_drop_menu.sort_index(inplace=True)
collection_df_drop_menu = collection_df_drop_menu.astype('float32')

test_date_df = pd.read_csv('forecast_date_and_meal_df.csv')  # 바로 읽어들이면 아마 날짜-식사명 중복이 존재할 것이다. nan끼리 group by sum하면 0이 되버려서 미리 뺴뒀는데 이렇게 되버림. 여기서 중복제거를 할 수 밖에 없다.
test_date_df = test_date_df.drop(columns=['Unnamed: 0'])
test_date_df = meal_index_encode(test_date_df)  # 이 시점에서 컬럼은 일자, 식사명, 수량 이 있고 식사명을 인코딩.
test_date_df = test_date_df.set_index(['일자', '식사명'])
test_date_df = test_date_df[~test_date_df.index.duplicated(keep='first')]  # group by sum하기 전에 미리 저장해서 중복된 test_date가 있다.
test_date_df.sort_index(inplace=True)
test_date_df = test_date_df.reset_index()

encodded_test_date_df = pd.DataFrame(data=list(test_date_df['식사명']), columns=['식사명Encoddeded'])
test_date_df = test_date_df.join(encodded_test_date_df)

test_date_df = pd.merge(test_date_df, collection_df_drop_menu.reset_index(),
                        on=['일자', '식사명'], how='inner').set_index(['일자', '식사명'])
outer_join_df = pd.merge(test_date_df, collection_df_drop_menu.reset_index(),
                        on=['일자', '식사명'], how='outer').set_index(['일자', '식사명'])
test_date_df = test_date_df.drop(columns=['수량_x', '수량_y', '식사명Encoddeded_x'])
# cols = test_date_df.columns.tolist()
# cols = cols[1:] + cols[:1]
# test_date_df = test_date_df[cols]
test_date_df = test_date_df.rename(columns={'식사명Encoddeded_y': '식사명Encoddeded'})
test_date_df = test_date_df.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
cols = collection_df_drop_menu.columns.tolist()

TrainXdf_scaled = scaler.fit_transform(collection_df_drop_menu.values[:, 1:])  # 수량 파트를 inverse transform하기 번거로우므로 뗏다 붙여서 normalization을 X에만 적용
mealDemand = collection_df_drop_menu.values[:, 0].reshape((-1, 1))
TrainXdf_scaled = np.concatenate((TrainXdf_scaled, mealDemand), axis=1)
TrainXdf = pd.DataFrame(data=TrainXdf_scaled, index=collection_df_drop_menu.index, columns=cols[1:] + cols[:1])  # 최종적으로 사용.
# TrainYdf = pd.DataFrame(data=collection_df_drop_menu.values[:, 0], index=collection_df_drop_menu.index, columns=[collection_df_drop_menu.columns[0]])
# Y값은 TrainXdf 0열에 있다. create dataset 함수 때문에 이렇게 한다.
TestXdf_scaled = scaler.fit_transform(test_date_df.values)
TestXdf = pd.DataFrame(data=TestXdf_scaled, index=test_date_df.index, columns=test_date_df.columns)  # 최종적으로 사용.

scriptName = os.path.basename(os.path.realpath(sys.argv[0]))

rmse_Scores = []

# hyper param
number_of_var = len(cols)
first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)
# print("first_layer_node_cnt %d" % first_layer_node_cnt)
epochs = 50
# epochs = 2  # stack으로 epoch2에 2 iteration 1분정도 걸림. 50iterataion이면 25분 여기에 epoch 50이면 1250분. 20시간 예상.
patience_num = 10
# patience_num = 2
look_back = 4 * 8  # test date 날짜 차이가 최소 8일(20140606과 20140529사이) 정도 되는 것 같다. 공백과 공백 사이로는 5일(점심2가 없는 날은 다행이 20110912 외엔 없으므로 20칸 정도차이)
# 번거롭고 별로 예측력이 강해질 것 같지도 않으니 8일전 자료(예측 포함)를 기반으로 계산. # 아마 점심2가 없는 20110912, 20120930 때문에 결과값이 하나 더나와서 삐뚤어지는 결과가 생길것이다.
forecast_ahead = 1  # 예측하는 건 일단 바로 다음의 끼니(다소 애매하지만 점심식사2는 특별.)
# forecast_ahead = 4 * 3  # 애초에 3일 뒤 것을 맞추는 문제다.
forecast_satck = []
###################

StartTrainDate = dt.strptime(str(20090803), '%Y%m%d').date() + timedelta(days=1)  # 20090803은 데이터 결락이 없는 마지막 날. 2010년 부터 추측해 제출하면 되므로 이게 더 좋을 것이다.
test_only_date = test_date_df.index.levels[0].tolist()
for num in range(0, len(test_only_date), 3):  # 50회 루프가 있을 것이다.
    print("----------------loop Num : ", num/3)
    # 최소 validation에 3일, tarin에 1일이라고 치면 가장 처음으로 빈칸이 생긴 날로부터 5일전이 최소로 필요. 물론 너무 적은 데이터라서 이 파트는 문제.
    # 일단 굴려보고 어떻게든 채워넣는 방법을 쓰는것도 나쁘진 않을 것 같다. !시간도 데이터도 부족하니 이 방법으로 간다.!
    # dt.strptime(str(test_only_date[num]), '%Y%m%d').date()는 최초로 빈칸이 생기는 날짜. +2 하면 제출일자가 된다.
    firstEmptyDate = dt.strptime(str(test_only_date[num]), '%Y%m%d').date()
    StartTrainDate = firstEmptyDate - timedelta(days=90)  # 다음 루프때 쓸 train data 구간 규정
    EndTrainDate = firstEmptyDate - timedelta(days=4)  # 20100709 # 2014-06-04 - 4 = 2010-05-30까지
    StartValidationDate = firstEmptyDate - timedelta(days=3)  # 20100710  # 20100705. 3일을 예측해야하지만 그전에 8일정도의 데이터를 기반으로 추측해서 RMSE를 추측할 예정.
    EndValidationDate = firstEmptyDate - timedelta(days=1)  # 20100712 : 20100705와는 8일차이.
    StartTestDate = firstEmptyDate  # 20100713
    EndTestDate = StartTestDate + timedelta(days=2)  # 20100715

    X_train_df = TrainXdf.loc[int(changeDateToStr(StartTrainDate)):int(changeDateToStr(EndValidationDate))]  # list slice와 달리 : 뒤쪽 항도 포함된다. # 일단 validataion 데이터도 같이 부른다.
    X_train = X_train_df.values
    # X_train = X_train_df.values[:, 22:]

    X_val_df = TrainXdf.loc[int(changeDateToStr(StartValidationDate)):int(changeDateToStr(EndValidationDate))]  # validation에 사용할 빈칸 이전의 3일관련 데이터.
    X_test_df = TestXdf.loc[int(changeDateToStr(StartTestDate)):int(changeDateToStr(EndTestDate))]
    blankCounta = X_test_df.shape[0]
    # if blankCounta != 12:
    #     print(X_val_df)

    # print("blankCounta : ", blankCounta)
    # print(X_train.shape)
    X_val = X_train[-look_back-blankCounta:, ]  # validation에 사용할 데이터 세트를 구축하기 위해 look_back 덧붙이기. 4*8 + 12
    X_train = X_train[:-blankCounta, ]  # validation할때 사용할 데이터만 제외하고 훈련

    print("StartTrainDate : ", StartTrainDate)
    print("EndTrainDate : ", EndTrainDate)
    # print(X_train.shape)
    print("StartValidationDate : ", StartValidationDate)
    print("EndValidationDate : ", EndValidationDate)
    # print(X_val.shape)
    print("StartTestDate : ", StartTestDate)
    print("EndTestDate : ", EndTestDate)

    X_train, y_train = create_dataset(X_train, look_back)
    X_val, y_val = create_dataset(X_val, look_back)
    # reshape training into [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], look_back, X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], look_back, X_val.shape[2])

    n_batch = gcd(X_train.shape[0], X_val.shape[0])  # 일단 배치사이즈를 대충 결정.
    model = Sequential()
    model.add(LSTM(number_of_var, batch_input_shape=(1, look_back, number_of_var), stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(number_of_var, batch_input_shape=(1, look_back, number_of_var), stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(number_of_var, batch_input_shape=(1, look_back, number_of_var), stateful=True))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 모델 저장 폴더 만들기
    MODEL_DIR = './' + scriptName + ' model_loopNum' + str(num).zfill(2) + '/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR + "{val_loss:.9f}.hdf5"
    # # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
    # 학습 자동 중단 설정
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    custom_hist = CustomHistory()
    custom_hist.init()
    # fit network # 에포크 2로 해서 14바퀴 돌리니까 93분정도 걸림. 다 돌려면 332분 정도 들텐데 이거에 25배면 138시간 든다. 고작 epoch 50에 약 6일 소요. 시간부족. 단어는 무시하는 걸로 한정시키자. stateless로 한다던가 해야할듯.
    for i in range(epochs):
        model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False, validation_data=(X_val, y_val),
                  callbacks=[custom_hist, checkpointer])
        model.reset_states()

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName+' model1_loopNum'+str(num).zfill(2))
    plt.plot(custom_hist.train_loss)
    plt.plot(custom_hist.val_loss)
    x_len = np.arange(len(custom_hist.val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    del model
    model = search_best_model(MODEL_DIR)
    evalScore = model.evaluate(X_val, y_val, batch_size=1)
    rmse_Scores.append(math.sqrt(evalScore))
    print("rmse : ", math.sqrt(evalScore))

    # 생각 같아선 추가된 데이터(validation 파트)를 포함해서 좀 더 훈련시켜두고 싶지만 마땅히 validation할 데이터가 없어 과적합 되기 십상. 그냥 이 모델 써서 예상해본다.
    for i in range(epochs):
        model.fit(X_val, y_val, epochs=1, batch_size=1, verbose=0, shuffle=False, validation_data=(X_train, y_train),
                  callbacks=[custom_hist, checkpointer])
        model.reset_states()

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName+' model2_loopNum'+str(num).zfill(2))
    plt.plot(custom_hist.train_loss)
    plt.plot(custom_hist.val_loss)
    plt.ylim(0.0, 50.0)
    x_len = np.arange(len(custom_hist.val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    del model
    model = search_best_model(MODEL_DIR)

    X_test = X_test_df.values
    x_hat = X_train[-1:, ]  # 끄트머리에서 다른 변수들로 계산.
    # x_hat = X_train[-look_back:, ]  # 끄트머리에서 다른 변수들로 계산.
    testPredict = np.zeros((blankCounta, 1))
    test_start_area_absolute_position = TrainXdf.index.get_loc(int(changeDateToStr(StartTestDate))).start
    for k in range(blankCounta):
        prediction = model.predict(x_hat, batch_size=n_batch)
        # prediction = model.predict(np.array([x_hat]), batch_size=1)
        testPredict[k] = prediction
        # new_x_hat = np.append([X_test[k, ], prediction])
        new_x_hat = np.vstack([np.reshape(X_test[k], (-1, 1)), prediction])
        x_hat = np.asarray([np.vstack([x_hat[0][1:], np.reshape(new_x_hat, (1, -1))])])
        # print(x_hat.shape)
        TrainXdf.iloc[test_start_area_absolute_position + k, -1] = prediction
    print("about  rmse: %s" % rmse_Scores)

    forecast_satck.extend(testPredict)
    # if num == 3:  # 마지막 루프만 아니면
    #     # StartTrainDate = EndTestDate + timedelta(days=1)  # 다음 루프때 쓸 train data 구간 규정
    #     break

    m, s = divmod((time.time() - start_time), 60)
    print("almost %d minute" % m)


print("\n about  rmse: %s" % rmse_Scores)
print("mean rmse %.7f:" % np.mean(rmse_Scores))

onlyDateInTest_df = list(set(list(test_date_df.index.levels[0])))
onlyDateInTest_df.sort()
onlyDateInTest_df = [onlyDateInTest_df[date_idx] for date_idx in range(2, len(onlyDateInTest_df), 3)]
prediction_for_meal_demand = pd.DataFrame(data=forecast_satck, index=test_date_df.index)
prediction_for_meal_demand = prediction_for_meal_demand.loc[onlyDateInTest_df]
prediction_for_meal_demand = prediction_for_meal_demand.round(1)
prediction_for_meal_demand = prediction_for_meal_demand.values.reshape((-1, 4))

prediction_for_meal_demand_df = pd.DataFrame(data=prediction_for_meal_demand, index=onlyDateInTest_df, columns=['아침식사', '점심식사', '점심식사2', '저녁식사'])
prediction_for_meal_demand_df.to_csv(scriptName + ' prediction.csv', encoding='utf-8')

"""
"""


