# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, sys
from collections import OrderedDict, defaultdict
from datetime import datetime as dt
from datetime import timedelta
from itertools import combinations

filename = os.getcwd() + '\swell_for_preprocess.csv'
swell_for_preprocess = pd.read_csv(filename, index_col=[0])
# abnormal_data = list(set(swell_for_preprocess.index.values))

filtered_swell_data = swell_for_preprocess[swell_for_preprocess['swell_happen'] == 1]  # swell이 일어난 구간 추출. index.value가 순수하게 swell이 일어난 날짜.

WallPo_swell = pd.read_csv('WallPo_swell.csv', index_col=[1])
GuRyoungPo_swell = pd.read_csv('GuRyoungPo_swell.csv', index_col=[1])
PoHang_weather14to17 = pd.read_csv('pohang_weather14to17.csv', index_col=[1])
# PoHang_weather14to17 = pd.read_csv('pohang_weather14to17_not_filterNA.csv', index_col=[1])
WallPo_swell = WallPo_swell.drop(columns=['지점'])
GuRyoungPo_swell = GuRyoungPo_swell.drop(columns=['지점'])
PoHang_weather14to17 = PoHang_weather14to17.drop(columns=['지점'])

# 점수 비중을 생각하면 1(swell일어난):2(swell일어나지 않은)training 셋 구성이 괜찮을 것이다. validation도 이 안에서 folding으로 랜덤하게 해볼 것.
# 30분 짜리도 있긴 하지만 딱히 정밀도에 영향을 주거나 할 것 같진 않다.

zeroMatrix = np.zeros(((365 * 4) + 1, 24)).astype('int')  # +1은 2016년 윤달. 7시부터 다음날 7시까지 24칸. 2014년 부터 2017년 자료를 대상. 1461row
date2014to2017 = list(PoHang_weather14to17.index.values)

dates_list = [dt.strptime(date, '%Y-%m-%d').date() for date in date2014to2017]
years = []
months = []
weekdays = []
weeknums = []
for date in dates_list:
    years.append(date.year)
    months.append(date.month)
    weekdays.append(date.weekday() + 1)  # 월요일이 1. 일요일이 7
    weeknums.append(date.isocalendar()[1])
data_about_time = pd.DataFrame({'year': years, 'month': months, 'weekday': weekdays, 'weeknums': weeknums},
                               index=date2014to2017)

swell_Y_DF = pd.DataFrame(data=zeroMatrix, index=date2014to2017)
time_grid = ["7:00", "8:00", "9:00", "10:00", "11:00", "12:00",
             "13:00", "14:00", "15:00", "16:00", "17:00", "18:00",
             "19:00", "20:00", "21:00", "22:00", "23:00", "0:00",
             "1:00", "2:00", "3:00", "4:00", "5:00", "6:00"]  # 시작시간 기준으로 일단 잡아본다.
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

swell_Y = []  # 2진법이라 치고 숫자로 바꿀까 생각해봤는데 무리수가 될 가능성도 있으니 나중으로 미뤄본다.
for index, row in swell_Y_DF.iterrows():
    binaries = "".join(str(i) for i in list(row.values))
    binaryToInt = int(binaries, 2)
    # print("binaryToInt %d" % binaryToInt)
    swell_Y.append(binaryToInt)
swell_Y = pd.DataFrame(data=swell_Y, index=date2014to2017)
swell_Y.to_csv('swell_Y_to_integer.csv', encoding='utf-8')
# "{0:b}".format(정수) 치면 string으로 다시 재 해석되어 나온다.

abnormal_date = pd.DataFrame(data=zeroMatrix, index=date2014to2017)  # 이상기후 있는 시점(너울성파도 포함)을 코딩. 독립변수 중 하나가 될수 있을까?
abnormal_date.columns = time_grid
for index, row in swell_for_preprocess.iterrows():
    # print(index)  # 이상기후이 생긴 날짜다. 근데 정작 test해야할 곳에선 이 정보가 있는지 없는지 애매하다.
    start_time = int(list(row)[4].split(':')[0]) - 7  # 마이너스로 참조하면 맨 뒤부터 하게 됨. 01~02경우 1-7=-6이 됨.
    # print('start_time_index %d' % start_time)
    # print(time_grid[start_time])
    for time in range(math.ceil(row['합'])):
        # print('time %d' % time)
        abnormal_date.loc[index, time_grid[start_time + time]] = 1
abnormal_date.to_csv('abnormal_date.csv', encoding='utf-8')
# X_with_date_and_abnormal_weather = pd.merge(abnormal_date, data_about_time, left_index=True, right_index=True)
# X_with_date_and_abnormal_weather.to_csv('X_with_date_and_abnormal_weather.csv', encoding='utf-8')

# PoHang_weather14to17['기사'].values
# Pohang_weather_24hour = pd.DataFrame(data=zeroMatrix, index=date2014to2017, columns=time_grid)
# for index, row in PoHang_weather14to17['기사'].iterrows():
#     start_time = list(row)[4].split(':')[0]

only_swell_date_data = set(filtered_swell_data.index.values)  # swell이 있었던 날만 찾으면 196일. 20150925 이전은 79일. 이후는 117일.
only_abnormal_date_data = set(swell_for_preprocess.index.values)  # 일단 이상기후(swell 포함) 있었던 날짜. # 571일이 나와야할텐데.
only_abnormal_date_data_without_swell = list(
    only_abnormal_date_data - only_swell_date_data)  # swell이 아닌 이상 기후만 있었던 날짜. 375일 20150925 이전은 177. 이후는 198일.
only_abnormal_date_data_without_swell = pd.DataFrame(data=only_abnormal_date_data_without_swell,
                                                     columns=['only_abnormal_date_data_without_swell'])
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
ommitted_date_of_GuRyoungPo_swell_date2014to2017 = set(date2014to2017) - set(GuRyoungPo_swell.index)
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
ommitted_date_of_WallPo_swell = set(date2014to2017) - set(WallPo_swell.index)
ommitted_date_of_WallPo_swell_after_20150925 = set([x for x in ommitted_date_of_WallPo_swell if x >= '2015-09-25'])
# 사실 월포 데이터는 dropna하면 26일자부터 데이터가 있게 된다. 그래서 이렇게 조건식을 달아도 큰 문제는 없다.
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
lack_of_GuRyoungPo_swell_info_date_at_test_after_20150925 = lack_of_GuRyoungPo_swell_info_date_at_test - set(
    lack_of_GuRyoungPo_swell_info_date_at_test_before_20150925)
lack_of_info_after_20150925 = lack_of_WallPo_info_date_at_test.union(
    lack_of_GuRyoungPo_swell_info_date_at_test_after_20150925)
print("lack_of_WallPo_and_GuRyoungPo_swell_info_date_After_20150925 : %s" % lack_of_info_after_20150925)
print(len(lack_of_info_after_20150925))

# 테스트에 필요한 데이터가 이만큼 없다.
PoHang_weather14to17 = PoHang_weather14to17.dropna()  # NA없애도 이미 처리해놔서 딱히 문제는 없다.
# PoHang_weather14to17_ommit = set(date2014to2017)-set(PoHang_weather14to17.index.values)

# independent_var_after20150925_1 = pd.merge(data_about_time, PoHang_weather14to17, WallPo_swell, GuRyoungPo_swell, left_index=True, right_index=True)
independent_var_with = pd.concat([data_about_time, PoHang_weather14to17], axis=1, join='inner')  # (1461, 34)
independent_var_with_Gu = pd.concat([data_about_time, PoHang_weather14to17, GuRyoungPo_swell], axis=1,
                                    join='inner')  # shape는 (1382, 43)
independent_var_with_Wall = pd.concat([data_about_time, PoHang_weather14to17, WallPo_swell], axis=1,
                                      join='inner')  # shape는 (792, 43)
independent_var_with_Gu_and_Wall = pd.concat([data_about_time, PoHang_weather14to17, GuRyoungPo_swell, WallPo_swell],
                                             axis=1, join='inner')  # shape는 (742, 52)

independent_var_only_pohang = pd.concat([data_about_time, PoHang_weather14to17], axis=1, join='inner')
independent_var_only_Gu = pd.concat([data_about_time, GuRyoungPo_swell], axis=1, join='inner')
independent_var_only_Wall = pd.concat([data_about_time, WallPo_swell], axis=1, join='inner')
independent_var_only_GuANdWall = pd.concat([data_about_time, GuRyoungPo_swell, WallPo_swell], axis=1, join='inner')

notInTestdates = test_dates - set(independent_var_with.index.values)
print(notInTestdates)  # 당연히 빈건 없다.
print(len(notInTestdates))
independent_var_with.to_csv('independent_var_with.csv', encoding='utf-8')  # 이걸 쓸 필요가 있을까?
independent_var_only_pohang.to_csv('independent_var_only_pohang.csv', encoding='utf-8')  # 이걸 쓸 필요가 있을까?

notInTestdates_Gu = test_dates - set(independent_var_with_Gu.index.values)
print(notInTestdates_Gu)
print(len(notInTestdates_Gu))
# 적어도 제출 할때 {'2015-12-13', '2016-10-20'} 빼고 부족한 건 없다. 이 둘은 월포 까지만 포함된 데이터 셋으로 훈련된 모델로 예측이 가능할 것 이다.
independent_var_with_Gu.to_csv('independent_var_with_Gu.csv', encoding='utf-8')
independent_var_only_Gu.to_csv('independent_var_only_Gu.csv', encoding='utf-8')  # 이걸 쓸 필요가 있을까?

notInTestdates_Wall = test_dates - set(independent_var_with_Wall.index.values)
print(notInTestdates_Wall)
print(len(notInTestdates_Wall))
# {'2015-01-13', '2014-09-25', '2017-03-15', '2015-04-04', '2014-05-18', '2016-03-04', '2014-12-21', '2014-10-23', '2015-07-18', '2015-06-27', '2014-07-06'}가 부족. 이 11개 날짜는 구룡포까지만 포함된 데이터로 예측.
independent_var_with_Wall.to_csv('independent_var_with_Wall.csv', encoding='utf-8')
independent_var_only_Wall.to_csv('independent_var_only_Wall.csv', encoding='utf-8')  # 이걸 쓸 필요가 있을까?

notInTestdates_Gu_and_Wall = test_dates - set(independent_var_with_Gu_and_Wall.index.values)
print(notInTestdates_Gu_and_Wall)
print(len(notInTestdates_Gu_and_Wall))
# {'2015-12-13', '2015-06-27', '2014-10-23', '2016-03-04', '2014-05-18', '2014-07-06', '2014-09-25', '2015-01-13', '2014-12-21', '2015-04-04', '2017-03-15', '2015-07-18', '2016-10-20'}가 부족. 위의 두 세트의 합집합이다.
# 이 데이터세트론 25개중 13개나 예측을 못하게 될 것이다.
independent_var_with_Gu_and_Wall.to_csv('independent_var_with_Gu_and_Wall.csv', encoding='utf-8')
independent_var_only_GuANdWall.to_csv('independent_var_only_GuANdWall.csv', encoding='utf-8')  # 이걸 쓸 필요가 있을까?

normal_date = set(date2014to2017) - set(only_abnormal_date_data) - test_dates  # only_abnormal_date_data는 swell 포함.
normal_date = list(normal_date)
normal_date = pd.DataFrame(data=normal_date, columns=['normal_date'])
normal_date.to_csv('normal_date.csv', encoding='utf-8')

only_swell_date_data = list(only_swell_date_data)
only_swell_date_data = pd.DataFrame(data=only_swell_date_data, columns=['swell_date'])
only_swell_date_data.to_csv('only_swell_date_data.csv', encoding='utf-8')

# 각 파트마다 정보량이 최대한 반영된 반큼의 모델만 쓰던가, 정보량이 최대로 반영된 만큼의 모델 앙상블 해야할 것 같다.

print('########################---------------flatten된 데이터 가공-------------------#########################')


def rotate(l, n):
    return l[-n:] + l[:-n]


rotated_time_grid = rotate(time_grid, 7)
# 이 아래 추가적인 데이터들의 index들은 7시를 기준으로 하는 하루 구성이 아니다. 일단 7칸 옮겨야할 것이다.

flatten_swell = swell_Y_DF.values.flatten().tolist()
index_for_flatten_swell = []
at_first = 0
for day in date2014to2017:
    if at_first == 0:
        for time in time_grid:
            index_for_flatten_swell.append(day + " " + time)
            if time == '23:00':
                at_first = 1
                break
    elif at_first == 1:
        for time in rotated_time_grid:
            index_for_flatten_swell.append(day + " " + time)

        if date2014to2017[-1] == day and time == '23:00':
            for idx in range(7):
                index_for_flatten_swell.append("2018-01-01 " + rotated_time_grid[idx])
                # print(index_for_flatten_swell[-1])
swell_Y_DF_flatten = pd.DataFrame(data=flatten_swell, index=index_for_flatten_swell)
swell_Y_DF_flatten.to_csv('swell_Y_DF_flatten.csv', encoding='utf-8')
# 함부로 정렬하면 9:00이 11:00 보다 커지게 된다. s = mydatetime.strftime('%Y-%m-%d %I:%M%p').lstrip("0").replace(" 0", " ")
# https://stackoverflow.com/questions/9525944/python-datetime-formatting-without-zero-padding  참고
test_dates_times = []
for test_day in test_dates:
    for test_time in time_grid:
        test_dates_times.append(test_day + " " + test_time)
test_dates_times_df = pd.DataFrame(data=test_dates_times, columns=['test_date'])
test_dates_times_df.to_csv('test_dates_times.csv', encoding='utf-8')

flatten_abnormal = abnormal_date.values.flatten().tolist()
abnormal_time_DF_flatten = pd.DataFrame(data=flatten_abnormal, index=index_for_flatten_swell)
only_abnormal_not_swell_time_DF_flatten = abnormal_time_DF_flatten - swell_Y_DF_flatten
only_abnormal_not_swell_time_DF_flatten.to_csv('only_abnormal_not_swell_time_DF_flatten.csv', encoding='utf-8')

GuRyoungPo_hour_14 = pd.read_csv('구룡포 시간별 파고부이 14년.csv', index_col=[1])
GuRyoungPo_hour_14 = GuRyoungPo_hour_14.drop(columns=['지점'])
GuRyoungPo_hour_15 = pd.read_csv('구룡포 시간별 파고부이 15년.csv', index_col=[1])
GuRyoungPo_hour_15 = GuRyoungPo_hour_15.drop(columns=['지점'])
GuRyoungPo_hour_16 = pd.read_csv('구룡포 시간별 파고부이 16년.csv', index_col=[1])
GuRyoungPo_hour_16 = GuRyoungPo_hour_16.drop(columns=['지점'])
GuRyoungPo_hour_17 = pd.read_csv('구룡포 시간별 파고부이 17년.csv', index_col=[1])
GuRyoungPo_hour_17 = GuRyoungPo_hour_17.drop(columns=['지점'])

WallPo_hour_15 = pd.read_csv('월포 시간별 파고부이 15년.csv', index_col=[1])
WallPo_hour_15 = WallPo_hour_15.drop(columns=['지점'])
WallPo_hour_16 = pd.read_csv('월포 시간별 파고부이 16년.csv', index_col=[1])
WallPo_hour_16 = WallPo_hour_16.drop(columns=['지점'])
WallPo_hour_17 = pd.read_csv('월포 시간별 파고부이 17년.csv', index_col=[1])
WallPo_hour_17 = WallPo_hour_17.drop(columns=['지점'])

Pohang_hour_14 = pd.read_csv('포항 시간별 해양기상부이 14년.csv', index_col=[1])
Pohang_hour_14 = Pohang_hour_14.drop(columns=['지점'])
Pohang_hour_15 = pd.read_csv('포항 시간별 해양기상부이 15년.csv', index_col=[1])
Pohang_hour_15 = Pohang_hour_15.drop(columns=['지점'])
Pohang_hour_16 = pd.read_csv('포항 시간별 해양기상부이 16년.csv', index_col=[1])
Pohang_hour_16 = Pohang_hour_16.drop(columns=['지점'])
Pohang_hour_17 = pd.read_csv('포항 시간별 해양기상부이 17년.csv', index_col=[1])
Pohang_hour_17 = Pohang_hour_17.drop(columns=['지점'])

GuRyoungPo_hour = pd.concat([GuRyoungPo_hour_14, GuRyoungPo_hour_15, GuRyoungPo_hour_16, GuRyoungPo_hour_17])
WallPo_hour = pd.concat([WallPo_hour_15, WallPo_hour_16, WallPo_hour_17])
Pohang_hour = pd.concat([Pohang_hour_14, Pohang_hour_15, Pohang_hour_16, Pohang_hour_17])

GuRyoungPo_hour = GuRyoungPo_hour.fillna(method='ffill', limit=1)
# print(GuRyoungPo_hour.isnull().sum())
GuRyoungPo_hour = GuRyoungPo_hour.dropna()
GuRyoungPo_hour_ommited_time_in_test_dates = list(
    set(test_dates_times) - set(GuRyoungPo_hour.index.values))  # 37개정도가 제출할 날짜에 부족. 전부 2015년 이후다.
GuRyoungPo_hour_ommited_time_in_test_dates.sort()
print("GuRyoungPo_hour_ommited_time_in_test_dates : ")  # 부족한건 월포의 정보로 로 때우던가 해야할 것이다.
print(GuRyoungPo_hour_ommited_time_in_test_dates)
print(len(GuRyoungPo_hour_ommited_time_in_test_dates))
# temp_list = ['2015-06-27 5:00']
# temp_df = pd.DataFrame(data=GuRyoungPo_hour.loc['2015-06-27 4:00'].values, columns=GuRyoungPo_hour.columns.values, index=temp_list)
# 20150925이전 날짜에 딱 한칸 부족한건 이전 시간의 데이터로 메운다.
temp_df = GuRyoungPo_hour.loc['2015-06-27 4:00']
temp_df.name = '2015-06-27 5:00'
GuRyoungPo_hour = GuRyoungPo_hour.append(temp_df)  # 이렇게하면 36줄 데이터가 부족하고 2015-12-13 24줄과 2016-10-20의 12시 이후 데이터만 부족. 2일치만 예측이 불가능.
# GuRyoungPo_hour.sort_index(inplace=True)

WallPo_hour = WallPo_hour.fillna(method='ffill', limit=1)
WallPo_hour = WallPo_hour.dropna()
WallPo_hour_ommited_time_in_test_dates = list(
    set(test_dates_times) - set(WallPo_hour.index.values))  # 2014년 데이터가 없어 누락이 많다. 263개
WallPo_hour_ommited_time_in_test_dates.sort()
WallPo_hour_ommited_time_in_test_dates_after_20150925_1000 = list(filter(lambda x: x > '2015-09-25 10:00', WallPo_hour_ommited_time_in_test_dates))
# print("WallPo_hour_ommited_time_in_test_dates : %s" % WallPo_hour_ommited_time_in_test_dates)
print("list of WallPo info lack date after 20150925 10:00 : ")
print(WallPo_hour_ommited_time_in_test_dates_after_20150925_1000)
print(len(WallPo_hour_ommited_time_in_test_dates_after_20150925_1000))  # 분기점 이후에는 47개 부족
# '2015-11-13 12:00', '2015-12-13 19:00', '2016-12-26 10:00', '2017-10-23 22:00' 만처리하면 2016-03-04, 2017-03-15(0~4시까지 빔)

temp_list_for_fill_one_section_before_1hour = ['2015-11-13 11:00', '2015-12-13 18:00', '2016-12-26 9:00', '2017-10-23 21:00']  # 이걸 불러
temp_list_for_fill_one_section = ['2015-11-13 12:00', '2015-12-13 19:00', '2016-12-26 10:00', '2017-10-23 22:00']  # 이걸로 바꾼다
# temp_df_list = pd.DataFrame()
for i in range(len(temp_list_for_fill_one_section_before_1hour)):
    temp_df = WallPo_hour.loc[temp_list_for_fill_one_section_before_1hour[i]]
    temp_df.name = temp_list_for_fill_one_section[i]
    # print(temp_df)
    WallPo_hour = WallPo_hour.append(temp_df)

WallPo_hour_ommited_time_in_test_dates = list(set(test_dates_times) - set(WallPo_hour.index.values))
WallPo_hour_ommited_time_in_test_dates_after_20150925_1000 = list(filter(lambda x: x > '2015-09-25 10:00', WallPo_hour_ommited_time_in_test_dates))
# '2015-09-25 10:00'는 '2015-09-25 9:00'보다 작아서 문제가 될 수 있으나 사실 이 이전의 데이터는 존재하지 않아 큰 문제는 없다.

GuWall_info_lack_time = set(WallPo_hour_ommited_time_in_test_dates_after_20150925_1000).intersection(
    GuRyoungPo_hour_ommited_time_in_test_dates)
print("GuWall_info_lack_time : %s" % GuWall_info_lack_time)  # 원래는 꽤 비는 부분이 많았으나 결국 줄이고 줄여서 둘 다 없는 부분은 일단 제거됨.

temp_df_list = pd.DataFrame(index=index_for_flatten_swell)
# temp_df_list = pd.concat([temp_df_list, Pohang_hour], axis=1, join='outer')

Pohang_hour_merged = pd.merge(temp_df_list, Pohang_hour, left_index=True, right_index=True, how='outer')
Pohang_hour = Pohang_hour_merged.fillna(method='ffill', limit=1)
Pohang_hour_without_wind = Pohang_hour.drop(columns=['풍속(m/s)', '풍향(deg)', 'GUST풍속(m/s)'])  # 풍속(1415), 풍향(1252), GUST풍속(1338)은 너무 빈게 많다. 사실상 이것 때문에 test_date에 없는게 많다.

Pohang_hour = Pohang_hour.dropna()
Pohang_hour_ommited_time_in_test_dates = list((set(test_dates_times) - set(Pohang_hour.index.values)))
Pohang_hour_ommited_time_in_test_dates.sort()
print("Pohang_hour_ommited_time_in_test_dates : ")
print(Pohang_hour_ommited_time_in_test_dates)
print(len(Pohang_hour_ommited_time_in_test_dates))  # 94개정도 부족. 총

Pohang_hour_without_wind = Pohang_hour_without_wind.dropna()
Pohang_hour_without_wind_ommited_time_in_test_dates = list(
    (set(test_dates_times) - set(Pohang_hour_without_wind.index.values)))
Pohang_hour_without_wind_ommited_time_in_test_dates.sort()
print("Pohang_hour_without_wind_ommited_time_in_test_dates : ")
print(Pohang_hour_without_wind_ommited_time_in_test_dates)
print(len(Pohang_hour_without_wind_ommited_time_in_test_dates))  # 53개정도 부족

# GuRyoungPo_hour.isnull().sum()
GuRyoungPo_hour.to_csv('GuRyoungPo_hour.csv', encoding='utf-8')
# WallPo_hour.isnull().sum()
WallPo_hour.to_csv('WallPo_hour.csv', encoding='utf-8')
# Pohang_hour.isnull().sum()
Pohang_hour.to_csv('Pohang_hour.csv', encoding='utf-8')
# Pohang_hour_without_wind.isnull().sum()
Pohang_hour_without_wind.to_csv('Pohang_hour_without_wind.csv', encoding='utf-8')

index_for_flatten_swell_date_list = [dt.strptime(flatten_index, '%Y-%m-%d %H:%M') for flatten_index in index_for_flatten_swell]
# 우선 date 자료형으로 변형
years = []  # 년도별 특성을 보정하기 위해
months = []  # 계절과 같은 요소를 표현하기 위해.
weekdays = []  # weeknums를 보조하기 위해
weeknums = []  # 1년에 따른 시간별
oClock = []  # 시간별 변화(낮 밤 같은 것이 있을 수 있으므로.
for flatten_index in index_for_flatten_swell_date_list:
    years.append(flatten_index.year)
    months.append(flatten_index.month)
    weekdays.append(flatten_index.weekday())  # 일요일이 0, 토요일이 7
    weeknums.append(flatten_index.isocalendar()[1])
    oClock.append(flatten_index.hour)
data_about_time_flatten = pd.DataFrame(
    {'year': years, 'month': months, 'weekday': weekdays, 'weeknums': weeknums, 'oClock': oClock},
    index=index_for_flatten_swell)
data_about_time_flatten.to_csv('data_about_time_flatten.csv', encoding='utf-8')  # 시간에 따른 영향을 측정시 사용.
'''
comb = combinations(['data_about_time_flatten', 'GuRyoungPo_hour', 'WallPo_hour', 'Pohang_hour'], 2)
for i in list(comb):
    print(i)
'''

# 4개 조합 : 2개 (포항은 크게 2가지)
ind_var_with_DateGuWallPo = pd.concat([data_about_time_flatten, GuRyoungPo_hour, WallPo_hour, Pohang_hour], axis=1, join='inner')
ind_var_with_DateGuWallPo.to_csv('ind_var_with_DateGuWallPo.csv', encoding='utf-8')
ind_var_with_DateGuWallPo_withoutwind = pd.concat([data_about_time_flatten, GuRyoungPo_hour, WallPo_hour, Pohang_hour_without_wind], axis=1, join='inner')
ind_var_with_DateGuWallPo_withoutwind.to_csv('ind_var_with_DateGuWallPo_withoutwind.csv', encoding='utf-8')

# 3개 조합 : 4*2개
ind_var_with_DateGuWall = pd.concat([data_about_time_flatten, GuRyoungPo_hour, WallPo_hour], axis=1, join='inner')
ind_var_with_DateGuWall.to_csv('ind_var_with_DateGuWall.csv', encoding='utf-8')
ind_var_with_DateGuPo = pd.concat([data_about_time_flatten, GuRyoungPo_hour, Pohang_hour], axis=1, join='inner')
ind_var_with_DateGuPo.to_csv('ind_var_with_DateGuPo.csv', encoding='utf-8')
ind_var_with_DateWallPo = pd.concat([data_about_time_flatten, WallPo_hour, Pohang_hour], axis=1, join='inner')
ind_var_with_DateWallPo.to_csv('ind_var_with_DateWallPo.csv', encoding='utf-8')
ind_var_with_GuWallPo = pd.concat([GuRyoungPo_hour, WallPo_hour, Pohang_hour], axis=1, join='inner')
ind_var_with_GuWallPo.to_csv('ind_var_with_GuWallPo.csv', encoding='utf-8')

ind_var_with_DateGuWall = pd.concat([data_about_time_flatten, GuRyoungPo_hour, WallPo_hour], axis=1, join='inner')
ind_var_with_DateGuWall.to_csv('ind_var_with_DateGuWall.csv', encoding='utf-8')
ind_var_with_DateGuPo_withoutwind = pd.concat([data_about_time_flatten, GuRyoungPo_hour, Pohang_hour_without_wind], axis=1, join='inner')
ind_var_with_DateGuPo_withoutwind.to_csv('ind_var_with_DateGuPo_withoutwind.csv', encoding='utf-8')
ind_var_with_DateWallPo_withoutwind = pd.concat([data_about_time_flatten, WallPo_hour, Pohang_hour_without_wind], axis=1, join='inner')
ind_var_with_DateWallPo_withoutwind.to_csv('ind_var_with_DateWallPo_withoutwind.csv', encoding='utf-8')
ind_var_with_GuWallPo_withoutwind = pd.concat([GuRyoungPo_hour, WallPo_hour, Pohang_hour_without_wind], axis=1, join='inner')
ind_var_with_GuWallPo_withoutwind.to_csv('ind_var_with_GuWallPo_withoutwind.csv', encoding='utf-8')

# 2개 조합 : 6개
ind_var_with_DateGu = pd.concat([data_about_time_flatten, GuRyoungPo_hour], axis=1, join='inner')
ind_var_with_DateGu.to_csv('ind_var_with_DateGu.csv', encoding='utf-8')
ind_var_with_DateWall = pd.concat([data_about_time_flatten, WallPo_hour], axis=1, join='inner')
ind_var_with_DateWall.to_csv('ind_var_with_DateWall.csv', encoding='utf-8')
ind_var_with_DatePo = pd.concat([data_about_time_flatten, Pohang_hour], axis=1, join='inner')
ind_var_with_DatePo.to_csv('ind_var_with_DatePo.csv', encoding='utf-8')
ind_var_with_GuWall = pd.concat([GuRyoungPo_hour, WallPo_hour], axis=1, join='inner')
ind_var_with_GuWall.to_csv('ind_var_with_GuWall.csv', encoding='utf-8')
ind_var_with_GuPo = pd.concat([GuRyoungPo_hour, Pohang_hour], axis=1, join='inner')
ind_var_with_GuPo.to_csv('ind_var_with_GuPo.csv', encoding='utf-8')
ind_var_with_WallPo = pd.concat([WallPo_hour, Pohang_hour], axis=1, join='inner')
ind_var_with_WallPo.to_csv('ind_var_with_WallPo.csv', encoding='utf-8')

ind_var_with_DateGu = pd.concat([data_about_time_flatten, GuRyoungPo_hour], axis=1, join='inner')
ind_var_with_DateGu.to_csv('ind_var_with_DateGu.csv', encoding='utf-8')
ind_var_with_DateWall = pd.concat([data_about_time_flatten, WallPo_hour], axis=1, join='inner')
ind_var_with_DateWall.to_csv('ind_var_with_DateWall.csv', encoding='utf-8')
ind_var_with_DatePo_withoutwind = pd.concat([data_about_time_flatten, Pohang_hour_without_wind], axis=1, join='inner')
ind_var_with_DatePo_withoutwind.to_csv('ind_var_with_DatePo_withoutwind.csv', encoding='utf-8')
ind_var_with_GuWall = pd.concat([GuRyoungPo_hour, WallPo_hour], axis=1, join='inner')
ind_var_with_GuWall.to_csv('ind_var_with_GuWall.csv', encoding='utf-8')
ind_var_with_GuPo_withoutwind = pd.concat([GuRyoungPo_hour, Pohang_hour_without_wind], axis=1, join='inner')
ind_var_with_GuPo_withoutwind.to_csv('ind_var_with_GuPo_withoutwind.csv', encoding='utf-8')
ind_var_with_WallPo_withoutwind = pd.concat([WallPo_hour, Pohang_hour_without_wind], axis=1, join='inner')
ind_var_with_WallPo_withoutwind.to_csv('ind_var_with_WallPo_withoutwind.csv', encoding='utf-8')

# 1개 조합 : 4개. 이미 위에서 한거다.



