#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, sys
from collections import OrderedDict, defaultdict
from datetime import datetime as dt

filename = os.getcwd() + '\swell_for_preprocess.csv'
# filename = os.getcwd() + '\deep_code\dataset\swell_for_preprocess.csv'
swell_for_preprocess = pd.read_csv(filename, index_col=[0])
# abnormal_data = list(set(swell_for_preprocess.index.values))

filtered_swell_data = swell_for_preprocess[swell_for_preprocess['swell_happen'] == 1]  # swell이 일어난 구간 추출. index.value가 순수하게 swell이 일어난 날짜.

# WallPo_swell = pd.read_csv(os.getcwd() +'\deep_code\dataset\WallPo_swell.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
# GuRyoungPo_swell = pd.read_csv(os.getcwd() +'\deep_code\dataset\GuRyoungPo_swell.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
# PoHang_weather14to17 = pd.read_csv(os.getcwd() +'\deep_code\dataset\PoHang_weather14to17.csv', index_col=[1])  # 독립변수 중 하나가 될것이다.
WallPo_swell = pd.read_csv('WallPo_swell.csv', index_col=[1])
GuRyoungPo_swell = pd.read_csv('GuRyoungPo_swell.csv', index_col=[1])
PoHang_weather14to17 = pd.read_csv('PoHang_weather14to17.csv', index_col=[1])
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
    # print("binaryToInt %d" % binaryToInt)
    swell_Y.append(binaryToInt)
swell_Y = pd.DataFrame(data=swell_Y, index=date2014to2017)
swell_Y.to_csv('swell_Y_to_integer.csv', encoding='utf-8')
# "{0:b}".format(정수) 치면 string으로 다시 재 해석되어 나온다.

# abnormal_date = pd.DataFrame(data=zeroMatrix, index=date2014to2017)  # 이상기후 있는 시점(너울성파도 포함)을 코딩. 독립변수 중 하나가 될수 있을까?
# abnormal_date.columns = time_grid
# for index, row in swell_for_preprocess.iterrows():
#     # print(index)  # 이상기후이 생긴 날짜다. 근데 정작 test해야할 곳에선 이 정보가 있는지 없는지 애매하다.
#     start_time = int(list(row)[4].split(':')[0]) - 7  # 마이너스로 참조하면 맨 뒤부터 하게 됨. 01~02경우 1-7=-6이 됨.
#     # print('start_time_index %d' % start_time)
#     # print(time_grid[start_time])
#     for time in range(math.ceil(row['합'])):
#         # print('time %d' % time)
#         abnormal_date.loc[index, time_grid[start_time + time]] = 1
# abnormal_date.to_csv('abnormal_date.csv', encoding='utf-8')
# X_with_date_and_abnormal_weather = pd.merge(abnormal_date, data_about_time, left_index=True, right_index=True)
# X_with_date_and_abnormal_weather.to_csv('X_with_date_and_abnormal_weather.csv', encoding='utf-8')

only_swell_date_data = set(filtered_swell_data.index.values)  # swell이 있었던 날만 찾으면 196일. 20150925 이전은 79일. 이후는 117일.
only_abnormal_date_data = set(swell_for_preprocess.index.values)  # 일단 이상기후(swell 포함) 있었던 날짜. # 571일이 나와야할텐데.
only_abnormal_date_data_without_swell = list(only_abnormal_date_data - only_swell_date_data)  # swell이 아닌 이상 기후만 있었던 날짜. 375일 20150925 이전은 177. 이후는 198일.
only_abnormal_date_data_without_swell = pd.DataFrame(data=only_abnormal_date_data_without_swell,  columns=['only_abnormal_date_data_without_swell'])
only_abnormal_date_data_without_swell.to_csv('only_abnormal_date_data_without_swell.csv', encoding='utf-8')

test_dates = pd.read_csv('test_form.csv', usecols=[0], skiprows=[0, 1])
# test_dates = pd.read_csv(os.getcwd() +'\\deep_code\\dataset\\test_form.csv', usecols=[0], skiprows=[0, 1])
test_dates = test_dates.values.flatten().tolist()
test_dates = set(test_dates)  # 25일 예측해야함.
test_dates_before_20150925 = set([x for x in test_dates if x < '2015-09-25'])
test_dates_after_20150925 = set([x for x in test_dates if x > '2015-09-25'])
print("test_dates : %s " % test_dates)

# 구룡포 데이터 2014년 부터 2017사이 없는 부분 확인.
GuRyoungPo_swell = GuRyoungPo_swell.dropna()  # NA없애면 1429에서 1382로. 47개 손실.
ommitted_date_of_GuRyoungPo_swell_date2014to2017 = set(date2014to2017)-set(GuRyoungPo_swell.index)
# 구룡포 데이터가 14년에서 17년 사이 없는 날
print("ommitted_date_of_GuRyoungPo_swell at 2014to2017 : %s" % ommitted_date_of_GuRyoungPo_swell_date2014to2017)
print(len(ommitted_date_of_GuRyoungPo_swell_date2014to2017))
lack_of_GuRyoungPo_swell_info_date = ommitted_date_of_GuRyoungPo_swell_date2014to2017.intersection(only_swell_date_data)
# 구룡포 데이터가 14년에서 17년 사이 없는 날 중 swell이 있는 날
print("lack_of_GuRyoungPo_swell_info_date : %s" % lack_of_GuRyoungPo_swell_info_date)  # train 할 때 swell에 관해 부족할 부분.
print(len(lack_of_GuRyoungPo_swell_info_date))
lack_of_GuRyoungPo_swell_info_date_at_test = test_dates.intersection(ommitted_date_of_GuRyoungPo_swell_date2014to2017)
lack_of_GuRyoungPo_swell_info_date_at_test_before_20150925 = [x for x in lack_of_GuRyoungPo_swell_info_date_at_test if x < '2015-09-25']
# test data에 있는 날에 구룡포 데이터가 없는 날. 20150925 이전
print("lack_of_GuRyoungPo_info_date_before_20150925 at test : %s" % lack_of_GuRyoungPo_swell_info_date_at_test_before_20150925)  # test 할 때 20150925이전에 데이터 부족은 없을것 같다.


# 월포 데이터 2014년 부터 2017사이 없는 부분 확인.
WallPo_swell = WallPo_swell.dropna()  # (2015년 9월 25일 부터 데이터가 존재함. row 818) NA 없애면 20150926 부터 row 792
ommitted_date_of_WallPo_swell = set(date2014to2017)-set(WallPo_swell.index)
ommitted_date_of_WallPo_swell_after_20150925 = set([x for x in ommitted_date_of_WallPo_swell if x >= '2015-09-25'])  # 사실 월포 데이터는 dropna하면 26일자부터 데이터가 있게 된다. 그래서 이렇게 조건식을 달아도 큰 문제는 없다.
# 월포 데이터가 2014년에서 2017년 사이 없는 날. 물론 2015년 9월 25일 이후
print("ommitted_date_of_WallPo_swell at 2014to2017 after 20150925 : %s" % ommitted_date_of_WallPo_swell_after_20150925)
print(len(ommitted_date_of_WallPo_swell_after_20150925))
lack_of_WallPo_swell_info_date = ommitted_date_of_WallPo_swell_after_20150925.intersection(only_swell_date_data)
# 월포 데이터가 2014년에서 2017년 사이(20150925이후) 없는 날 중 swell이 있는 날
print("lack_of_WallPo_swell_info_date : %s" % lack_of_WallPo_swell_info_date)
print(len(lack_of_WallPo_swell_info_date))
lack_of_WallPo_info_date_at_test = test_dates.intersection(ommitted_date_of_WallPo_swell_after_20150925)
# test data에 있는 날에 월포 데이터가 없는 날. 20150925 이후
print("lack_of_WallPo_info_date at test : %s" % lack_of_WallPo_info_date_at_test)  # test 할 때 20150925이후에 정보 부족은 없을것 같다.

# 20150925 이후 구룡포 또는 월포 데이터가 없는 날짜.
lack_of_GuRyoungPo_swell_info_date_at_test_after_20150925 = lack_of_GuRyoungPo_swell_info_date_at_test - set(lack_of_GuRyoungPo_swell_info_date_at_test_before_20150925)
lack_of_info_after_20150925 = lack_of_WallPo_info_date_at_test.union(lack_of_GuRyoungPo_swell_info_date_at_test_after_20150925)
print("lack_of_WallPo_and_GuRyoungPo_swell_info_date_After_20150925 : %s" % lack_of_info_after_20150925)
print(len(lack_of_info_after_20150925))


# 테스트에 필요한 데이터가 이만큼 없다.
PoHang_weather14to17 = PoHang_weather14to17.dropna()  # NA없애도 이미 처리해놔서 딱히 문제는 없다.
# PoHang_weather14to17_ommit = set(date2014to2017)-set(PoHang_weather14to17.index.values)

# independent_var_after20150925_1 = pd.merge(data_about_time, PoHang_weather14to17, WallPo_swell, GuRyoungPo_swell, left_index=True, right_index=True)
independent_var_with = pd.concat([data_about_time, PoHang_weather14to17], axis=1, join='inner')  # (1461, 34)
independent_var_with_Gu = pd.concat([data_about_time, PoHang_weather14to17, GuRyoungPo_swell], axis=1, join='inner')  # shape는 (1382, 43)
independent_var_with_Wall = pd.concat([data_about_time, PoHang_weather14to17, WallPo_swell], axis=1, join='inner')  # shape는 (792, 43)
independent_var_with_Gu_and_Wall = pd.concat([data_about_time, PoHang_weather14to17, GuRyoungPo_swell, WallPo_swell], axis=1, join='inner')  # shape는 (742, 52)

notInTestdates = test_dates - set(independent_var_with.index.values)
print(notInTestdates)  # 당연히 빈건 없다.
print(len(notInTestdates))
independent_var_with.to_csv('independent_var_with.csv', encoding='utf-8')  # 이걸 쓸 필요가 있을까?

notInTestdates_Gu = test_dates - set(independent_var_with_Gu.index.values)
print(notInTestdates_Gu)
print(len(notInTestdates_Gu))
# 적어도 제출 할때 {'2015-12-13', '2016-10-20'} 빼고 부족한 건 없다. 이 둘은 월포 까지만 포함된 데이터 셋으로 훈련된 모델로 예측이 가능할 것 이다.
independent_var_with_Gu.to_csv('independent_var_with_Gu.csv', encoding='utf-8')

notInTestdates_Wall = test_dates - set(independent_var_with_Wall.index.values)
print(notInTestdates_Wall)
print(len(notInTestdates_Wall))
# {'2015-01-13', '2014-09-25', '2017-03-15', '2015-04-04', '2014-05-18', '2016-03-04', '2014-12-21', '2014-10-23', '2015-07-18', '2015-06-27', '2014-07-06'}가 부족. 이 11개 날짜는 구룡포까지만 포함된 데이터로 예측.
independent_var_with_Wall.to_csv('independent_var_with_Wall.csv', encoding='utf-8')

notInTestdates_Gu_and_Wall = test_dates - set(independent_var_with_Gu_and_Wall.index.values)
print(notInTestdates_Gu_and_Wall)
print(len(notInTestdates_Gu_and_Wall))
# {'2015-12-13', '2015-06-27', '2014-10-23', '2016-03-04', '2014-05-18', '2014-07-06', '2014-09-25', '2015-01-13', '2014-12-21', '2015-04-04', '2017-03-15', '2015-07-18', '2016-10-20'}가 부족. 위의 두 세트의 합집합이다.
# 이 데이터세트론 25개중 13개나 예측을 못하게 될 것이다.
independent_var_with_Gu_and_Wall.to_csv('independent_var_with_Gu_and_Wall.csv', encoding='utf-8')

normal_date = set(date2014to2017) - set(only_abnormal_date_data) - test_dates  # only_abnormal_date_data는 swell 포함.
normal_date = list(normal_date)
normal_date = pd.DataFrame(data=normal_date,  columns=['normal_date'])
normal_date.to_csv('normal_date.csv', encoding='utf-8')

only_swell_date_data = list(only_swell_date_data)
only_swell_date_data = pd.DataFrame(data=only_swell_date_data,  columns=['swell_date'])
only_swell_date_data.to_csv('only_swell_date_data.csv', encoding='utf-8')

# 각 파트마다 정보량이 최대한 반영된 반큼의 모델만 쓰던가, 정보량이 최대로 반영된 만큼의 모델 앙상블 해야할 것 같다.



