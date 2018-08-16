#-*- coding: utf-8 -*-
'''
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
'''
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from datetime import datetime as dt

import os, sys
import tensorflow as tf
import random as rn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
start_time = time.time()


def score_calculating(true_value, pred_value):
    Score = 0
    for i in range(len(true_value)):
        if true_value[i] == 0:
            if true_value[i] == pred_value[i]:
                Score = Score + 1
            else:
                Score = Score - 1
        else:
            # print(true_value[i][j], ":", pred_value[i][j], end=",")
            if true_value[i] == pred_value[i]:
                Score = Score + 2
            else:
                Score = Score - 2
    return Score


def time_index_change_format(df):
    as_list = df.index.tolist()
    for dateAndTime in as_list:
        position = as_list.index(dateAndTime)
        as_list[position] = dt.strptime(dateAndTime, '%Y-%m-%d %H:%M').strftime('%Y-%m-%d %H:%M')
    df.index = as_list
    return df


# fix random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
rn.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

test_dates_df = pd.read_csv('test_dates_times.csv', index_col=[1], skiprows=0)
test_dates_df = time_index_change_format(test_dates_df)
# test_dates_df.rename( columns={'Unnamed: 0':'new column name'}, inplace=True )
# test_dates_df.sort_index(inplace=True)
test_dates = test_dates_df.index.values.flatten().tolist()
# 데이터 불러오기
X_df = pd.read_csv('ind_var_with_DateGu.csv', index_col=[0])
X_df = time_index_change_format(X_df)
X_df.sort_index(inplace=True)

X_df_index = set(list(X_df.index.values)) - set(test_dates)  # 제출해야할 날짜는 우선적으로 뺀다.
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 측정일자와 데이터세트가 겹치는 시간.
abnormal_date = pd.read_csv('only_abnormal_not_swell_time_DF_flatten.csv', index_col=[0])
abnormal_date = time_index_change_format(abnormal_date)
abnormal_date.sort_index(inplace=True)
abnormal_date = abnormal_date[abnormal_date['0'] == 1].index.values
abnormal_date = set(abnormal_date).intersection(X_df_index)

swell_date = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
swell_date = time_index_change_format(swell_date)
swell_date.sort_index(inplace=True)
swell_date = swell_date[swell_date['0'] == 1].index.values
swell_date = set(swell_date).intersection(X_df_index)

# normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = (X_df_index-swell_date) - abnormal_date  # test도 swell도 비정상 날씨도 아닌 날.
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))


# 오버 샘플링 없이 모든 데이터 사용 : 그다지 예측력이 좋아진 느낌은 없다.
'''
# normal_date_X_df = X_df.loc[normal_date]
# abnormal_date_X_df = X_df.loc[abnormal_date]
# swell_date_X_df = X_df.loc[swell_date]
X_train_df = X_df.loc[X_df_index]

X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 비정상인날(swell제외) 전부 : swell이 일어나는 날. 대회에선 정상인 날자는 신경쓰지 않는다고 첫날에 가정. : 그다지 예측력이 좋아진 느낌은 없다.

abnormal_date_X_df = X_df.loc[abnormal_date]
swell_date_X_df = X_df.loc[swell_date]

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast_df = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast_df.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_df = time_index_change_format(Y_df)
onlyXnotY = set(X_df.index.values) - set(Y_df.index.values)
onlyYnotX = set(Y_df.index.values) - set(X_df.index.values)
Y_df.sort_index(inplace=True)
Y_df = Y_df.loc[set(X_train_df.index.values)]
# Y_df = Y_df.reindex(set(X_df.index.values))
Y = Y_df.values

# 날씨가 비정상인날(swell제외) 1 : swell이 일어나는 날 1 비율로 오버 샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 정상인날 1 : 날씨가 비정상인날(swell 제외) 1: swell이 일어나는 날 1 비율로 오버샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
normal_date_X_df = X_df.loc[normal_date].sample(len(swell_date))
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([normal_date_X_df, abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# swell로만 학습 : 형편없다. 나중에 결과를 합치는데 써봐야할 것이다.
'''
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)
X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values  # 24시간 100101011... 같은 형태의 Y값
'''

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=None)
y_test = y_test.flatten().tolist()
y_train = y_train.flatten().tolist()
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(y_pred.sum())
# print(np.asarray(predictions).sum())
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("scoring : %d" % score_calculating(y_test, predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(model.feature_importances_)
varList = []
for num in range(len(model.feature_importances_)):
    # varName = 'f' + str(num)
    # varList.append(varName)
    varList.append(num)
var_feature_imp = pd.DataFrame(data=varList, index=list(model.feature_importances_))
var_feature_imp.sort_index(inplace=True, ascending=False)
var_sortby_featureImp = list(var_feature_imp.values.flatten().tolist())
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plot_importance(model)
# plt.show()
scoreList = []
thresholds = np.sort(model.feature_importances_)
forecastList = []
accuracyList = []
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                                    max_depth=6, min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                    objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=42)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    Score = score_calculating(y_test, predictions)
    scoreList.append(Score)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%% , Score: %d, OneCount : %d"
          % (thresh, select_X_train.shape[1], accuracy*100.0, Score, np.asarray(predictions).sum()))
    accuracyList.append(accuracy*100.0)
    # y_fore = selection_model.predict( X_forecast_df.iloc[list(var_sortby_featureImp[:select_X_train.shape[1]])].values )
    y_fore = selection_model.predict(selection.transform(X_forecast))
    forecast = [round(value) for value in y_fore]
    forecastList.append(forecast)
    print("Onecount : %d " % np.asanyarray(forecast).sum())

print("mean score : %.4f " % np.mean(scoreList))

idx = scoreList.index(np.max(scoreList))
print("take %d th thresh hold is fine" % idx)
print("this is best acc prediction : %s" % forecastList[idx])
result = pd.DataFrame(data=forecastList[idx], index=X_forecast_df.index)
result = time_index_change_format(result)
result.sort_index(inplace=True)
# test_dates_df = test_dates_df.drop(columns=['Unnamed: 0'])

# Forecast_df = pd.DataFrame(index=test_dates_df.index.values)
Forecast_df = test_dates_df
for index, row in Forecast_df.iterrows():
    if index in result.index:
        Forecast_df.loc[index, 'Unnamed: 0'] = int(result.loc[index].values)
# Forecast_df = pd.concat([test_dates_df, result], axis=1, join='inner')
# df.filter(regex='a',axis=0)
# https://stackoverflow.com/questions/22897195/selecting-rows-with-similar-index-names-in-pandas






X_df = pd.read_csv('ind_var_with_DateWall.csv', index_col=[0])
X_df = time_index_change_format(X_df)
X_df.sort_index(inplace=True)

X_df_index = set(list(X_df.index.values)) - set(test_dates)  # 제출해야할 날짜는 우선적으로 뺀다.
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 측정일자와 데이터세트가 겹치는 시간.
abnormal_date = pd.read_csv('only_abnormal_not_swell_time_DF_flatten.csv', index_col=[0])
abnormal_date = time_index_change_format(abnormal_date)
abnormal_date.sort_index(inplace=True)
abnormal_date = abnormal_date[abnormal_date['0'] == 1].index.values
abnormal_date = set(abnormal_date).intersection(X_df_index)

swell_date = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
swell_date = time_index_change_format(swell_date)
swell_date.sort_index(inplace=True)
swell_date = swell_date[swell_date['0'] == 1].index.values
swell_date = set(swell_date).intersection(X_df_index)

# normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = (X_df_index-swell_date) - abnormal_date  # test도 swell도 비정상 날씨도 아닌 날.
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))


# 오버 샘플링 없이 모든 데이터 사용 : 그다지 예측력이 좋아진 느낌은 없다.
'''
# normal_date_X_df = X_df.loc[normal_date]
# abnormal_date_X_df = X_df.loc[abnormal_date]
# swell_date_X_df = X_df.loc[swell_date]
X_train_df = X_df.loc[X_df_index]

X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 비정상인날(swell제외) 전부 : swell이 일어나는 날. 대회에선 정상인 날자는 신경쓰지 않는다고 첫날에 가정. : 그다지 예측력이 좋아진 느낌은 없다.

abnormal_date_X_df = X_df.loc[abnormal_date]
swell_date_X_df = X_df.loc[swell_date]

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast_df = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast_df.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_df = time_index_change_format(Y_df)
onlyXnotY = set(X_df.index.values) - set(Y_df.index.values)
onlyYnotX = set(Y_df.index.values) - set(X_df.index.values)
Y_df.sort_index(inplace=True)
Y_df = Y_df.loc[set(X_train_df.index.values)]
# Y_df = Y_df.reindex(set(X_df.index.values))
Y = Y_df.values


# 날씨가 비정상인날(swell제외) 1 : swell이 일어나는 날 1 비율로 오버 샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 정상인날 1 : 날씨가 비정상인날(swell 제외) 1: swell이 일어나는 날 1 비율로 오버샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
normal_date_X_df = X_df.loc[normal_date].sample(len(swell_date))
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([normal_date_X_df, abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# swell로만 학습 : 형편없다. 나중에 결과를 합치는데 써봐야할 것이다.
'''
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)
X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values  # 24시간 100101011... 같은 형태의 Y값
'''

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=None)
y_test = y_test.flatten().tolist()
y_train = y_train.flatten().tolist()
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(y_pred.sum())
# print(np.asarray(predictions).sum())
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("scoring : %d" % score_calculating(y_test, predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(model.feature_importances_)
varList = []
for num in range(len(model.feature_importances_)):
    # varName = 'f' + str(num)
    # varList.append(varName)
    varList.append(num)
var_feature_imp = pd.DataFrame(data=varList, index=list(model.feature_importances_))
var_feature_imp.sort_index(inplace=True, ascending=False)
var_sortby_featureImp = list(var_feature_imp.values.flatten().tolist())
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plot_importance(model)
# plt.show()
scoreList = []
thresholds = np.sort(model.feature_importances_)
forecastList = []
accuracyList = []
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                                    max_depth=6, min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                    objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=42)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    Score = score_calculating(y_test, predictions)
    scoreList.append(Score)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%% , Score: %d, OneCount : %d"
          % (thresh, select_X_train.shape[1], accuracy*100.0, Score, np.asarray(predictions).sum()))
    accuracyList.append(accuracy*100.0)
    # y_fore = selection_model.predict( X_forecast_df.iloc[list(var_sortby_featureImp[:select_X_train.shape[1]])].values )
    y_fore = selection_model.predict(selection.transform(X_forecast))
    forecast = [round(value) for value in y_fore]
    forecastList.append(forecast)
    print("Onecount : %d " % np.asanyarray(forecast).sum())

print("mean score : %.4f " % np.mean(scoreList))

idx = scoreList.index(np.max(scoreList))
print("take %d th thresh hold is fine" % idx)
print("this is best acc prediction : %s" % forecastList[idx])
result = pd.DataFrame(data=forecastList[idx], index=X_forecast_df.index)
result = time_index_change_format(result)
result.sort_index(inplace=True)

for index, row in Forecast_df.iterrows():
    if index in result.index:
        Forecast_df.loc[index] = int(result.loc[index].values)
# Forecast_df = pd.concat([Forecast_df, result], axis=1, join='inner')

# df.filter(regex='a',axis=0)
# https://stackoverflow.com/questions/22897195/selecting-rows-with-similar-index-names-in-pandas














X_df = pd.read_csv('ind_var_with_DateGuWall.csv', index_col=[0])
X_df = time_index_change_format(X_df)
X_df.sort_index(inplace=True)

X_df_index = set(list(X_df.index.values)) - set(test_dates)  # 제출해야할 날짜는 우선적으로 뺀다.
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 측정일자와 데이터세트가 겹치는 시간.
abnormal_date = pd.read_csv('only_abnormal_not_swell_time_DF_flatten.csv', index_col=[0])
abnormal_date = time_index_change_format(abnormal_date)
abnormal_date.sort_index(inplace=True)
abnormal_date = abnormal_date[abnormal_date['0'] == 1].index.values
abnormal_date = set(abnormal_date).intersection(X_df_index)

swell_date = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
swell_date = time_index_change_format(swell_date)
swell_date.sort_index(inplace=True)
swell_date = swell_date[swell_date['0'] == 1].index.values
swell_date = set(swell_date).intersection(X_df_index)

# normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = (X_df_index-swell_date) - abnormal_date  # test도 swell도 비정상 날씨도 아닌 날.
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))


# 오버 샘플링 없이 모든 데이터 사용 : 그다지 예측력이 좋아진 느낌은 없다.
'''
# normal_date_X_df = X_df.loc[normal_date]
# abnormal_date_X_df = X_df.loc[abnormal_date]
# swell_date_X_df = X_df.loc[swell_date]
X_train_df = X_df.loc[X_df_index]

X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 비정상인날(swell제외) 전부 : swell이 일어나는 날. 대회에선 정상인 날자는 신경쓰지 않는다고 첫날에 가정. : 그다지 예측력이 좋아진 느낌은 없다.

abnormal_date_X_df = X_df.loc[abnormal_date]
swell_date_X_df = X_df.loc[swell_date]

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast_df = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast_df.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_df = time_index_change_format(Y_df)
onlyXnotY = set(X_df.index.values) - set(Y_df.index.values)
onlyYnotX = set(Y_df.index.values) - set(X_df.index.values)
Y_df.sort_index(inplace=True)
Y_df = Y_df.loc[set(X_train_df.index.values)]
# Y_df = Y_df.reindex(set(X_df.index.values))
Y = Y_df.values

# 날씨가 비정상인날(swell제외) 1 : swell이 일어나는 날 1 비율로 오버 샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 정상인날 1 : 날씨가 비정상인날(swell 제외) 1: swell이 일어나는 날 1 비율로 오버샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
normal_date_X_df = X_df.loc[normal_date].sample(len(swell_date))
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([normal_date_X_df, abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# swell로만 학습 : 형편없다. 나중에 결과를 합치는데 써봐야할 것이다.
'''
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)
X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values  # 24시간 100101011... 같은 형태의 Y값
'''

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=None)
y_test = y_test.flatten().tolist()
y_train = y_train.flatten().tolist()
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(y_pred.sum())
# print(np.asarray(predictions).sum())
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("scoring : %d" % score_calculating(y_test, predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(model.feature_importances_)
varList = []
for num in range(len(model.feature_importances_)):
    # varName = 'f' + str(num)
    # varList.append(varName)
    varList.append(num)
var_feature_imp = pd.DataFrame(data=varList, index=list(model.feature_importances_))
var_feature_imp.sort_index(inplace=True, ascending=False)
var_sortby_featureImp = list(var_feature_imp.values.flatten().tolist())
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plot_importance(model)
# plt.show()
scoreList = []
thresholds = np.sort(model.feature_importances_)
forecastList = []
accuracyList = []
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                                    max_depth=6, min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                    objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=42)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    Score = score_calculating(y_test, predictions)
    scoreList.append(Score)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%% , Score: %d, OneCount : %d"
          % (thresh, select_X_train.shape[1], accuracy*100.0, Score, np.asarray(predictions).sum()))
    accuracyList.append(accuracy*100.0)
    # y_fore = selection_model.predict( X_forecast_df.iloc[list(var_sortby_featureImp[:select_X_train.shape[1]])].values )
    y_fore = selection_model.predict(selection.transform(X_forecast))
    forecast = [round(value) for value in y_fore]
    forecastList.append(forecast)
    print("Onecount : %d " % np.asanyarray(forecast).sum())

print("mean score : %.4f " % np.mean(scoreList))

idx = scoreList.index(np.max(scoreList))
print("take %d th thresh hold is fine" % idx)
print("this is best acc prediction : %s" % forecastList[idx])
result = pd.DataFrame(data=forecastList[idx], index=X_forecast_df.index)
result = time_index_change_format(result)
result.sort_index(inplace=True)

for index, row in Forecast_df.iterrows():
    if index in result.index:
        Forecast_df.loc[index] = int(result.loc[index].values)
# Forecast_df = pd.concat([Forecast_df, result], axis=1, join='inner')









X_df = pd.read_csv('ind_var_with_DateGuWallPo.csv', index_col=[0])
X_df = time_index_change_format(X_df)
X_df.sort_index(inplace=True)

X_df_index = set(list(X_df.index.values)) - set(test_dates)  # 제출해야할 날짜는 우선적으로 뺀다.
test_dates_in_X_df = set(test_dates).intersection(set(X_df.index.values))  # 측정일자와 데이터세트가 겹치는 시간.
abnormal_date = pd.read_csv('only_abnormal_not_swell_time_DF_flatten.csv', index_col=[0])
abnormal_date = time_index_change_format(abnormal_date)
abnormal_date.sort_index(inplace=True)
abnormal_date = abnormal_date[abnormal_date['0'] == 1].index.values
abnormal_date = set(abnormal_date).intersection(X_df_index)

swell_date = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
swell_date = time_index_change_format(swell_date)
swell_date.sort_index(inplace=True)
swell_date = swell_date[swell_date['0'] == 1].index.values
swell_date = set(swell_date).intersection(X_df_index)

# normal_date = pd.read_csv('normal_date.csv', index_col=[0]).values.flatten().tolist()
normal_date = (X_df_index-swell_date) - abnormal_date  # test도 swell도 비정상 날씨도 아닌 날.
print("length check normal : %d, abnormal : %d, swell : %d" % (len(normal_date), len(abnormal_date), len(swell_date)))


# 오버 샘플링 없이 모든 데이터 사용 : 그다지 예측력이 좋아진 느낌은 없다.
'''
# normal_date_X_df = X_df.loc[normal_date]
# abnormal_date_X_df = X_df.loc[abnormal_date]
# swell_date_X_df = X_df.loc[swell_date]
X_train_df = X_df.loc[X_df_index]

X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 비정상인날(swell제외) 전부 : swell이 일어나는 날. 대회에선 정상인 날자는 신경쓰지 않는다고 첫날에 가정. : 그다지 예측력이 좋아진 느낌은 없다.

abnormal_date_X_df = X_df.loc[abnormal_date]
swell_date_X_df = X_df.loc[swell_date]

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast_df = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast_df.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_df = time_index_change_format(Y_df)
onlyXnotY = set(X_df.index.values) - set(Y_df.index.values)
onlyYnotX = set(Y_df.index.values) - set(X_df.index.values)
Y_df.sort_index(inplace=True)
Y_df = Y_df.loc[set(X_train_df.index.values)]
# Y_df = Y_df.reindex(set(X_df.index.values))
Y = Y_df.values


# 날씨가 비정상인날(swell제외) 1 : swell이 일어나는 날 1 비율로 오버 샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# 날씨가 정상인날 1 : 날씨가 비정상인날(swell 제외) 1: swell이 일어나는 날 1 비율로 오버샘플링 : 그다지 예측력이 좋아진 느낌은 없다.
'''
normal_date_X_df = X_df.loc[normal_date].sample(len(swell_date))
abnormal_date_X_df = X_df.loc[abnormal_date].sample(len(swell_date))
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([normal_date_X_df, abnormal_date_X_df, swell_date_X_df])
X = X_train_df.values.astype('float32')
# X_scaler = MinMaxScaler(feature_range=(0, 1))
# X = X_scaler.fit_transform(X)

X_forecast = X_df.loc[test_dates_in_X_df]
X_forecast = X_forecast.values.astype('float32')

Y_df = pd.read_csv('swell_Y_DF_flatten.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values
'''
# swell로만 학습 : 형편없다. 나중에 결과를 합치는데 써봐야할 것이다.
'''
swell_date_X_df = X_df.loc[swell_date].sample(len(swell_date))

X_train_df = pd.concat([swell_date_X_df])
X = X_train_df.values.astype('float32')
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)
X_test = X_df.loc[test_dates_in_X_df]

Y_df = pd.read_csv('swell_Y.csv', index_col=[0])
Y_train_df = Y_df.loc[set(X_train_df.index.values)]
Y = Y_train_df.values  # 24시간 100101011... 같은 형태의 Y값
'''

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=None)
y_test = y_test.flatten().tolist()
y_train = y_train.flatten().tolist()
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(y_pred.sum())
# print(np.asarray(predictions).sum())
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("scoring : %d" % score_calculating(y_test, predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(model.feature_importances_)
varList = []
for num in range(len(model.feature_importances_)):
    # varName = 'f' + str(num)
    # varList.append(varName)
    varList.append(num)
var_feature_imp = pd.DataFrame(data=varList, index=list(model.feature_importances_))
var_feature_imp.sort_index(inplace=True, ascending=False)
var_sortby_featureImp = list(var_feature_imp.values.flatten().tolist())
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plot_importance(model)
# plt.show()
scoreList = []
thresholds = np.sort(model.feature_importances_)
forecastList = []
accuracyList = []
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                                    max_depth=6, min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                    objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=42)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    Score = score_calculating(y_test, predictions)
    scoreList.append(Score)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%% , Score: %d, OneCount : %d"
          % (thresh, select_X_train.shape[1], accuracy*100.0, Score, np.asarray(predictions).sum()))
    accuracyList.append(accuracy*100.0)
    # y_fore = selection_model.predict( X_forecast_df.iloc[list(var_sortby_featureImp[:select_X_train.shape[1]])].values )
    y_fore = selection_model.predict(selection.transform(X_forecast))
    forecast = [round(value) for value in y_fore]
    forecastList.append(forecast)
    print("Onecount : %d " % np.asanyarray(forecast).sum())

print("mean score : %.4f " % np.mean(scoreList))

idx = scoreList.index(np.max(scoreList))
print("take %d th thresh hold is fine" % idx)
print("this is best acc prediction : %s" % forecastList[idx])
result = pd.DataFrame(data=forecastList[idx], index=X_forecast_df.index)
result = time_index_change_format(result)
result.sort_index(inplace=True)
for index, row in Forecast_df.iterrows():
    if index in result.index:
        Forecast_df.loc[index] = int(result.loc[index].values)
# Forecast_df = pd.concat([Forecast_df, result], axis=1, join='inner')


test_dates_notTime_DF = pd.read_csv('test_form.csv', usecols=[0], skiprows=[0, 1])
test_dates_notTime = test_dates_notTime_DF.values.flatten().tolist()  # 제출해야할 날짜.

# for DATE in test_dates_notTime:
#     tempDF = Forecast_df.filter(regex=DATE, axis=0).T
#     # df.filter(regex='a',axis=0) # https://stackoverflow.com/questions/22897195/selecting-rows-with-similar-index-names-in-pandas
#     commitForm = pd.concat([commitForm, tempDF])

commitForm_dat = Forecast_df.values
commitForm_dat = np.reshape(commitForm_dat, (-1, 24))
commitForm = pd.DataFrame(data=commitForm_dat)
print(commitForm)
commitForm.to_csv('commitForm.csv', encoding='utf-8')