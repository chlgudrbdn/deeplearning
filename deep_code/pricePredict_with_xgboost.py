# -*- coding: utf-8 -*-

# plot feature importance manually
from numpy import loadtxt
# from numpy import genfromtxt
from numpy import sort
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import pandas

import time
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
start_time = time.time()
#------------------------

# sourceEncoding = "iso-8859-1"
# targetEncoding = "utf-8"
# source = open("source")
# target = open("target", "w")
#
# target.write(unicode(source.read(), sourceEncoding).encode(targetEncoding))

# load data
# dataset = loadtxt('../dataset/full_data_about_iron_ore.csv', delimiter=",", encoding="CP1252", skiprows=1)
# dataset = pandas.read_csv('../dataset/full_data_about_iron_ore.csv', encoding="EUC-KR").values
dataset = pandas.read_csv('../dataset/full_data_about_iron_ore.csv').values
# dataset = genfromtxt('../dataset/full_data_about_iron_ore-NA_not_processed.csv', encoding="utf-8", delimiter=",", skip_header=True)[:, 1:]
# split data into X and y
X = dataset[:, 1:]
Y = dataset[:, 0]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42) # 1/3을 테스트에 사용. random_state는 걍 시드 같다.

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# feature importance
# print(model.feature_importances_)
# plot
# pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

# plot_importance(model)
# pyplot.show()
pd=pandas.DataFrame(model.feature_importances_)
pd.to_csv("important_feature.csv")

#############
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = mean_squared_error(y_test, predictions)  # 만약 카테고리를 예측할거면 여기에 acurracy
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# # Fit model using each importance as a threshold
# thresholds = sort(model.feature_importances_)
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)  # 중요도에 따라 feature 선택하는 클래스.
#     select_X_train = selection.transform(X_train) # X_train의 선택된 feature를 줄인다.
#     # train model
#     selection_model = XGBClassifier()
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)
#     predictions = [round(value) for value in y_pred]
#     accuracy = mean_squared_error(y_test, predictions)
#     print("Thresh=%.3f, n=%d, RMSE: %.2f" % (thresh, select_X_train.shape[1], accuracy))
# Thresh=0.000, n=452, RMSE: 148.98

#----------------------------
#종료부분 코드
print("--- %s seconds ---" %(time.time() - start_time))