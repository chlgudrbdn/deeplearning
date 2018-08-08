#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, sys
from collections import OrderedDict, defaultdict
from datetime import datetime as dt

# filename = os.getcwd() + '\swell_for_preprocess.csv'
filename = os.getcwd() + '\dataset\swell_for_preprocess.csv'
swell_for_preprocess = pd.read_csv(filename, index_col=[0])
# abnormal_data = list(set(swell_for_preprocess.index.values))

filtered_swell_data = swell_for_preprocess[swell_for_preprocess['swell_happen'] == 1]

WallPo_swell = pd.read_csv(os.getcwd() +'\dataset\WallPo_swell.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
GuRyoungPo_swell = pd.read_csv(os.getcwd() +'\dataset\GuRyoungPo_swell.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
PoHang_weather14to17 = pd.read_csv(os.getcwd() +'\dataset\PoHang_weather14to17.csv', index_col=[1], )  # 독립변수 중 하나가 될것이다.
# WallPo_swell = pd.read_csv('WallPo_swell.csv')
# GuRyoungPo_swell = pd.read_csv('GuRyoungPo_swell.csv')
# PoHang_weather14to17 = pd.read_csv('PoHang_weather14to17.csv')

# 일단 오버샘플링도 고려할 필요는 있다만 그건 나중에 생각해보자.
# 30분 짜리도 있긴 하지만 딱히 정밀도에 영향을 주거나 할 것 같진 않다.

zeroMatrix = np.zeros(((365*4)+1, 24))  # +1은 2016년 윤달. 7시부터 다음날 7시까지 24칸. 2014년 부터 2017년 자료를 대상.
date2014to2017 = list(PoHang_weather14to17.index.values)

dates_list = [dt.strptime(date, '%Y-%m-%d').date() for date in date2014to2017]
years = []
months = []
weekdays = []
weeknums = []
for date in dates_list:
    years.append(date.year)
    months.append(date.month)
    weekdays.append(date.weekday()+1) # 월요일이 1. 일요일이 7
    weeknums.append(date.isocalendar()[1])
# years = [date.year for date in dates_list]
# months = [date.month for date in dates_list]
# weekdays = [date.weekday() for date in dates_list]
# weeknums = [date.today().isocalendar()[1] for in dates_list]
data_about_time = pd.DataFrame({'year': years, 'month': months, 'weekday': weekdays, 'weeknums': weeknums},
                               index=date2014to2017)

swell_Y_DF = pd.DataFrame(data=zeroMatrix, index=date2014to2017)
time_grid = ["07:00~08:00", "08:00~09:00", "09:00~10:00", "10:00~11:00", "11:00~12:00", "12:00~13:00",
             "13:00~14:00", "14:00~15:00", "15:00~16:00", "16:00~17:00", "17:00~18:00", "18:00~19:00",
             "19:00~20:00", "20:00~21:00", "21:00~22:00", "22:00~23:00", "23:00~24:00", "00:00~01:00",
             "01:00~02:00", "02:00~03:00", "03:00~04:00", "04:00~05:00", "05:00~06:00", "06:00~07:00"]
swell_Y_DF.columns = time_grid

for index, row in filtered_swell_data.iterrows():  # 훈련용, 테스트용 종속변수가 될 것이다.
    # print(index)  # 너울이 생긴 날짜다.
    start_time = int(list(row)[4].split(':')[0]) - 7  # 마이너스로 참조하면 맨 뒤부터 하게 됨. 01~02경우 1-7=-6이 됨.
    # print('start_time_index %d' % start_time)
    # print(time_grid[start_time])
    for time in range(math.ceil(row['합'])):
        # print('time %d' % time)
        swell_Y_DF.loc[index, time_grid[start_time + time]] = 1
swell_Y_DF.to_csv('swell_Y.csv', encoding='utf-8')
# swell_Y_for_test = pd.read_csv('swell_Y', index_col=[0])

abnormal_date = pd.DataFrame(data=zeroMatrix, index=date2014to2017)  # 이상기후 있는 시점(너울성파도 포함)을 코딩. 독립변수 중 하나가 될것이다.
abnormal_date.columns = time_grid
for index, row in swell_for_preprocess.iterrows():
    # print(index)  # 이상기후이 생긴 날짜다.
    start_time = int(list(row)[4].split(':')[0]) - 7  # 마이너스로 참조하면 맨 뒤부터 하게 됨. 01~02경우 1-7=-6이 됨.
    # print('start_time_index %d' % start_time)
    # print(time_grid[start_time])
    for time in range(math.ceil(row['합'])):
        # print('time %d' % time)
        abnormal_date.loc[index, time_grid[start_time + time]] = 1
abnormal_date.to_csv('abnormal_date.csv', encoding='utf-8')
X_with_date_and_abnormal_weather = pd.merge(abnormal_date, data_about_time, left_index=True, right_index=True)
X_with_date_and_abnormal_weather.to_csv('X_with_date_and_abnormal_weather .csv', encoding='utf-8')






ommitted_date_of_GuRyoungPo_swell = set(date2014to2017)-set(GuRyoungPo_swell.index)
print("ommitted_date_of_GuRyoungPo_swell : %s" % ommitted_date_of_GuRyoungPo_swell)
could_be_lack_of_GuRyoungPo_swell_info_date = ommitted_date_of_GuRyoungPo_swell.intersection(set(filtered_swell_data.index.values))
print("could_be_lack_of_GuRyoungPo_swell_info_date : %s" % could_be_lack_of_GuRyoungPo_swell_info_date)






WallPo_swell.drop(columns=['지점'])
GuRyoungPo_swell.drop(columns=['지점'])
PoHang_weather14to17.drop(columns=['지점'])
