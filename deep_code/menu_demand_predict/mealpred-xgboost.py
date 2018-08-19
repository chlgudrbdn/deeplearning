# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import plot_importance
from xgboost import XGBClassifier
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import random as rn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

import math

import time
start_time = time.time()

'''
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
'''

# fix random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
rn.seed(seed)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# tf.set_random_seed(seed)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# manual_variable_initialization(True)
# tf.global_variables_initializer()

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

Y = collection_values_float32[:, 0]
X = collection_values_float32[:, 1:]

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X_fore = scaler.fit_transform(test_date_df.values.astype('float32'))


number_of_var = X.shape[1]
first_layer_node_cnt = int(number_of_var*(number_of_var-1)/2)
n_fold = 10
kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

rmse_Scores = []

scriptName = os.path.basename(os.path.realpath(sys.argv[0]))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)
y_test = y_test.flatten().tolist()
y_train = y_train.flatten().tolist()
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
# evaluate predictions
mse = mean_squared_error(y_test, y_pred)
print("RMSE: %.9f%%" % (math.sqrt(mse)))
# print(model.feature_importances_)

# important한 변수 추적
varList = []
for num in range(len(model.feature_importances_)):
    varList.append(num)
var_feature_imp = pd.DataFrame(data=varList, index=list(model.feature_importances_))
var_feature_imp.sort_index(inplace=True, ascending=False)
var_sortby_featureImp = list(var_feature_imp.values.flatten().tolist())

# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plot_importance(model)
# plt.show()

thresholds = np.sort(model.feature_importances_)
forecastList = []  # blending에 쓸 수 있을지도?
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    # selection_model = XGBClassifier(learning_rate=0.1, n_estimators=1000,
    #                                 max_depth=6, min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
    #                                 objective='reg:linear', nthread=4, scale_pos_weight=1, seed=42)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Thresh=%.3f, n=%d, mse: %.2f%% , " % (thresh, select_X_train.shape[1], mse))
    rmse_Scores.append(math.sqrt(mse))
    # y_fore = selection_model.predict( X_forecast_df.iloc[list(var_sortby_featureImp[:select_X_train.shape[1]])].values )
    y_fore = selection_model.predict(selection.transform(X_fore))
    forecastList.append(y_fore)

print("rmse mean : %.4f " % np.mean(rmse_Scores))

idx = rmse_Scores.index(np.min(rmse_Scores))
print("take %d th thresh hold is fine" % idx)
print("this is best rmse prediction : %s" % rmse_Scores[idx])
result = pd.DataFrame(data=forecastList[idx], index=test_date_df.index)
result.sort_index(inplace=True)
print(result)

print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %d minute" % m)


'''
for train_index, validation_index in kf.split(X):  # 이하 모델을 학습한 뒤 테스트.
    print("loop num : ", len(rmse_Scores)+1)
    # print("TRAIN: %d" % len(train_index), "TEST: %d" % len(validation_index))
    X_train, X_Validation = X[train_index], X[validation_index]
    Y_train, Y_Validation = Y[train_index], Y[validation_index]
    model = Sequential()
    model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu'))
    edge_num = 2
    while int(first_layer_node_cnt * (edge_num**(-2))) >= 5 and edge_num < 6:
        model.add(Dense(int(first_layer_node_cnt * (edge_num**(-2))), activation='relu'))
        model.add(Dropout(0.1))
        edge_num += 1
    model.add(Dense(1))
    print("edge_num : %d" % edge_num)
    model.compile(loss='mse', optimizer='adam')
    # model.compile(loss='mse', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999), metrics=[rmse])

    # 모델 저장 폴더 만들기
    MODEL_DIR = './'+scriptName+' model_loopNum'+str(len(rmse_Scores)).zfill(2)+'/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    modelpath = MODEL_DIR+"{val_loss:.9f}.hdf5"
    # # 모델 업데이트 및 저장
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
    # 학습 자동 중단 설정
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_num)
    history = model.fit(X_train, Y_train, validation_data=(X_Validation, Y_Validation), epochs=epochs, verbose=0,
                        callbacks=[checkpointer, early_stopping_callback], batch_size=256)
    # history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, verbose=2, callbacks=[early_stopping_callback, checkpointer])

    plt.figure(figsize=(8, 8)).canvas.set_window_title(scriptName+' model_loopNum'+str(len(rmse_Scores)).zfill(2))
    # 테스트 셋의 오차
    # y_rmse = history.history['rmse']
    # y_vrmse = history.history['val_rmse']
    y_loss = history.history['loss']
    y_vloss = history.history['val_loss']
    # 그래프로 표현
    plt.ylim(0.0, 50.0)
    x_len = np.arange(len(y_loss))
    # plt.plot(x_len, y_rmse, c="blue", label='y_rmse')
    # plt.plot(x_len, y_vrmse, c="red", label='y_vrmse')
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
    model = load_model(MODEL_DIR + '{0:.9f}'.format(file_list[0]) + ".hdf5")
    evalScore = model.evaluate(X_Validation, Y_Validation, batch_size=len(X_Validation))

    # prediction_for_train = model.predict(X_train, batch_size=len(X_Validation))
    # prediction_for_val = model.predict(X_Validation, batch_size=len(X_Validation))
    # print(evalScore)
    # trainScore = math.sqrt(mean_squared_error(Y_train, prediction_for_train[:, 0]))
    # print('Train Score: %.9f RMSE' % trainScore)
    # trainScoreList.append(trainScore)
    # valScore = math.sqrt(mean_squared_error(Y_Validation, prediction_for_val[:, 0]))
    # print('Val Score: %.9f RMSE' % valScore)

    # print("predict : %s" % prediction_for_val)
    # print("real    : %s" % Y_Validation)
    rmse_Scores.append(math.sqrt(evalScore))

print("\n %d fold rmse: %s" % (n_fold, rmse_Scores))
# accuracy = [float(j) for j in rmse_Scores]
print("mean rmse %.7f:" % np.mean(rmse_Scores))

print("--- %s seconds ---" % (time.time() - start_time))
m, s = divmod((time.time() - start_time), 60)
print("almost %d minute" % m)
'''

'''
model = Sequential()
model.add(Dense(first_layer_node_cnt, input_dim=number_of_var, activation='relu'))
edge_num = 2
while int(first_layer_node_cnt * (edge_num ** (-2))) >= 5 and edge_num < 6:
    model.add(Dense(int(first_layer_node_cnt * (edge_num ** (-2))), activation='relu'))
    model.add(Dropout(0.1))
    edge_num += 1
model.add(Dense(1))
print("edge_num : %d" % edge_num)
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
                    callbacks=[checkpointer, early_stopping_callback], batch_size=len(X))

file_list = os.listdir(MODEL_DIR)  # 루프 가장 최고 모델 다시 불러오기.
file_list.sort()  # 만든날짜 정렬
model = load_model(MODEL_DIR + file_list[0])

evalScore = model.evaluate(X, Y, batch_size=len(X))

prediction_for_train = model.predict(X, batch_size=len(X))
trainScore = math.sqrt(mean_squared_error(Y, prediction_for_train[:, 0]))
print("evalScore", end=" ")
print(evalScore)  # print(evalScore)로 확인 결과 loss, rmse. 앞에걸 sqrt하면 그냥 일일이 계산한 결과와 같은게 튀어나오는 것 같다.
print('Train Score: %.9f RMSE' % trainScore)

prediction_for_test = model.predict(X_test, batch_size=len(X_test))

# for i in range(len(prediction_for_test)):
#     print("date: %s" % test_date_df.index.values[i][0], "meal: %s" % test_date_df.index.values[i][1],
#           "prediction: %.4f:" % prediction_for_test[i])
prediction_for_meal_demand = pd.DataFrame(data=prediction_for_test, index=test_date_df.index)
prediction_for_meal_demand.to_csv('prediction_for_test_dnn.csv', encoding='utf-8')
'''
