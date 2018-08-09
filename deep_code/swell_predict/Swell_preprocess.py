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
filename = os.getcwd() + '\deep_code\dataset\swell_for_preprocess.csv'
swell_for_preprocess = pd.read_csv(filename, index_col=[0])
# abnormal_data = list(set(swell_for_preprocess.index.values))

filtered_swell_data = swell_for_preprocess[swell_for_preprocess['swell_happen'] == 1]

WallPo_swell = pd.read_csv(os.getcwd() +'\deep_code\dataset\WallPo_swell.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
GuRyoungPo_swell = pd.read_csv(os.getcwd() +'\deep_code\dataset\GuRyoungPo_swell.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
PoHang_weather14to17 = pd.read_csv(os.getcwd() +'\deep_code\dataset\PoHang_weather14to17.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
# WallPo_swell = pd.read_csv('WallPo_swell.csv', index_col=[1])
# GuRyoungPo_swell = pd.read_csv('GuRyoungPo_swell.csv', index_col=[1])
# PoHang_weather14to17 = pd.read_csv('PoHang_weather14to17.csv', index_col=[1])
WallPo_swell = WallPo_swell.drop(columns=['지점'])
GuRyoungPo_swell = GuRyoungPo_swell.drop(columns=['지점'])
PoHang_weather14to17 = PoHang_weather14to17.drop(columns=['지점'])

# 점수 비중을 생각하면 1(swell일어난):2(swell일어나지 않은)training 셋 구성이 괜찮을 것이다. validation도 이 안에서 folding으로 랜덤하게 해볼 것.
# 30분 짜리도 있긴 하지만 딱히 정밀도에 영향을 주거나 할 것 같진 않다.

zeroMatrix = np.zeros(((365*4)+1, 24)).astype('int')  # +1은 2016년 윤달. 7시부터 다음날 7시까지 24칸. 2014년 부터 2017년 자료를 대상. 1461row
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

swell_Y = [] # 2진법이라 치고 숫자로 바꿀까 생각해봤는데 무리수가 될 가능성도 있으니 나중으로 미뤄본다.
for index, row in swell_Y_DF.iterrows():
    binaries = "".join(str(i) for i in list(row.values))
    binaryToInt = int(binaries, 2)
    print("binaryToInt %d" % binaryToInt)
    swell_Y.append(binaryToInt)
swell_Y = pd.DataFrame(data=swell_Y, index=date2014to2017)
swell_Y.to_csv('swell_Y_to_integer.csv', encoding='utf-8')
# "{0:b}".format(정수) 치면 string으로 다시 재 해석되어 나온다.


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
X_with_date_and_abnormal_weather.to_csv('X_with_date_and_abnormal_weather.csv', encoding='utf-8')

# 예측해야하는 구간에 데이터가 존재하는지 확인하기 위한 예측 구간 확인용 코드
# test_dates = pd.read_csv('test_form.csv', usecols=[0], skiprows=[0, 1])
test_dates = pd.read_csv(os.getcwd() +'\\deep_code\\dataset\\test_form.csv', usecols=[0], skiprows=[0, 1])
test_dates = test_dates.values.flatten().tolist()
test_dates = set(test_dates)  # 25일 예측해야함.

# 구룡포 데이터 2014년 부터 2017사이 없는 부분 확인.
ommitted_date_of_GuRyoungPo_swell = set(date2014to2017)-set(GuRyoungPo_swell.index)
print("ommitted_date_of_GuRyoungPo_swell at 2014to2017 : %s" % ommitted_date_of_GuRyoungPo_swell)
could_be_lack_of_GuRyoungPo_swell_info_date = ommitted_date_of_GuRyoungPo_swell.intersection(set(filtered_swell_data.index.values))
print("could_be_lack_of_GuRyoungPo_swell_info_date : %s" % could_be_lack_of_GuRyoungPo_swell_info_date)  # train 할 때 부족할 부분. 6개정도.
lack_of_GuRyoungPo_swell_info_date = test_dates.intersection(could_be_lack_of_GuRyoungPo_swell_info_date)
print("lack_of_GuRyoungPo_swell_info_date : %s" % test_dates.intersection(lack_of_GuRyoungPo_swell_info_date))  # test 할 때 데이터 부족은 없을것 같다.
GuRyoungPo_swell = GuRyoungPo_swell.dropna()  # NA없애면 row 1382

# 월포 데이터 2014년 부터 2017사이 없는 부분 확인. (2015년 9월 25일 부터 데이터가 존재함. row 818)
ommitted_date_of_WallPo_swell = set(date2014to2017)-set(WallPo_swell.index)
print("ommitted_date_of_WallPo_swell at 2014to2017 : %s" % ommitted_date_of_WallPo_swell)
could_be_lack_of_WallPo_swell_info_date = ommitted_date_of_WallPo_swell.intersection(set(filtered_swell_data.index.values))
print("could_be_lack_of_WallPo_swell_info_date : %s" % could_be_lack_of_WallPo_swell_info_date)
lack_of_WallPo_swell_info_date = test_dates.intersection(could_be_lack_of_WallPo_swell_info_date)
print("lack_of_WallPo_swell_info_date : %s" % test_dates.intersection(lack_of_WallPo_swell_info_date))  # test 할 때 정보 부족은 없을것 같다.
WallPo_swell = WallPo_swell.dropna()  # NA없애면 row 792

newPoHang_weather14to17 = PoHang_weather14to17.dropna()  # NA없애면 row 792