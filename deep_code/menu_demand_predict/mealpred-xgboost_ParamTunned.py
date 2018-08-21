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
print("RMSE: %.9f" % (math.sqrt(mse)))
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
    # selection_model = XGBClassifier()
    selection_model = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                                    max_depth=6, min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                    objective='reg:linear', nthread=4, scale_pos_weight=1, seed=seed)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Thresh=%.3f, n=%d, mse: %.2f , " % (thresh, select_X_train.shape[1], mse))
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

# 별로 유망하지 못한 결과다. 애써봐야 9를 못넘음.
# RMSE: 13.499226750%
# Thresh=0.000, n=28, mse: 86.24% ,
# Thresh=0.000, n=28, mse: 86.24% ,
# Thresh=0.000, n=28, mse: 86.24% ,
# Thresh=0.000, n=28, mse: 86.24% ,
# Thresh=0.000, n=24, mse: 91.78% ,
# Thresh=0.000, n=23, mse: 88.21% ,
# Thresh=0.002, n=22, mse: 91.04% ,
# Thresh=0.011, n=21, mse: 94.78% ,
# Thresh=0.012, n=20, mse: 87.77% ,
# Thresh=0.015, n=19, mse: 94.47% ,