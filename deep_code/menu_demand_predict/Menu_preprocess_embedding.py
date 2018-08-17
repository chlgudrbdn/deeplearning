#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, sys
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict, defaultdict
from keras.preprocessing import sequence
from collections import Counter

# filename = os.getcwd() + '\menu.csv'
menu_df = pd.read_csv('menu.csv', index_col=[1, 2])
menu_df = menu_df.drop(columns=['Unnamed: 0'])
menu_df_sum = menu_df.groupby(level=[0, 1]).sum()
# 여러 인덱스에서 같은 인덱스에 속한애들은 그룹핑해서 더하기.
# https://stackoverflow.com/questions/24826368/summing-over-a-multiindex-level-in-a-pandas-series
menu_arr = menu_df_sum.values   # ndarray 형태 각 식사(아침, 점심, 저녁, 점심2)별 메뉴

# 메뉴 추출해 번호 부여
menu_set = set(['깍두기(손칼)'])
all_menu_for_freq_count = []
for menu_per_meal in menu_arr:
    menu_elem = "".join(list(menu_per_meal)).split(",")
    menu_set = menu_set.union(set(menu_elem))  # 메뉴 총 개수 3075. 한끼에 최대 메뉴 19. 아침까지 합쳐야하니까 실제론 더 길것.
    all_menu_for_freq_count.extend(menu_elem)

# all_menu_for_freq_count = all_menu_for_freq_count.flatten()
counts = Counter(all_menu_for_freq_count)
counts.most_common()
# print(counts)

Menu_list = list(menu_set)
# Menu_list.sort()  # 일단 비슷한 것 끼리 숫자라도 가까우면 좋을 것 같다. 생각같아선 벡터를 쓰고 싶다.
# 빈도 같은걸 기반으로 하거나.
menu_dict = dict() # 메뉴별 빈도.
for num in range(len(Menu_list)):
    # print("num : %d" % num ,"menu_dict : %s" % menu_dict)
    menu_dict[counts.most_common()[num][0]] = num + 1  # 0부터 시작해서
    # dict(list(zip(Menu_list, range(len(Menu_list)))))

embedded_menu_list = []
MaxLength = 0
for menu_per_meal in menu_arr:
    menu_elem = "".join(list(menu_per_meal)).split(",")  # 끼니별 메뉴를 리스트화
    if len(menu_elem) >= MaxLength:  # 최대길이 구해두기
        MaxLength = len(menu_elem)
        # print(menu_elem)
    new_menu_elem = []
    for menuItem in menu_elem:  # 번호로 바꿀수 있도록 리스트 별로 바꿔둠.
        new_menu_elem.append(menu_dict[menuItem])
    embedded_menu_list.append(new_menu_elem)
print("MaxLength : %d" % MaxLength)
embedded_menu_list_padded = sequence.pad_sequences(embedded_menu_list, maxlen=MaxLength)
embedded_menu_list_padded_df = pd.DataFrame(data=embedded_menu_list_padded, index=menu_df_sum.index)

meal_demand_df = pd.read_csv('meal_demand.csv', index_col=[1, 2])
# meal_demand_df = meal_demand.loc[:, ~meal_demand.columns.str.contains('^Unnamed')]
meal_demand_df = meal_demand_df.drop(columns=['Unnamed: 0'])
# forecast_date_and_meal = [a for tuple_in_array in meal_demand_df[meal_demand_df['수량'].isna()].index]
forecast_date_and_meal = meal_demand_df[meal_demand_df['수량'].isna()].reset_index()
# forecast_date_and_meal = np.asarray(meal_demand_df[meal_demand_df['수량'].isna()].index)
forecast_date_and_meal_df = pd.DataFrame(data=forecast_date_and_meal)
forecast_date_and_meal_df.to_csv("forecast_date_and_meal_df.csv", encoding='utf-8')
meal_demand_df_sum = meal_demand_df.groupby(level=[0, 1]).sum()

collection_data = pd.merge(meal_demand_df_sum.reset_index(), menu_df_sum.reset_index(),
                           on=['일자', '식사명'], how='outer').set_index(['일자', '식사명'])
collection_data = pd.merge(collection_data.reset_index(), embedded_menu_list_padded_df.reset_index(),
                           on=['일자', '식사명'], how='outer').set_index(['일자', '식사명'])
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

collection_data.to_csv("collection_data.csv", encoding='utf-8')
# 전처리 과정중에 확인한 사실:
# 식사인원 시트에서 2004년 3월 데이터가 꽤 빠져 있다. 메뉴에 20040315~20040320에는 자료가 있지만 식사인원에는 그 사이 자료가 존재하지 않는다.
# 200704의 점심, 20081228 아점저, 20080715~20090726, 20090803도 마찬가지이다. 합해서 64건 정도.
# 데이터가 없어야할 예측해야하는 날짜는 group by sum 하는 과정에서 아예 0으로 처리된 듯 싶다. 일단 따로 빼두었으니 틀의 유지에는 문제가 없다.
# 데이터가 없는 고로 빠진 부분은 그냥 없애는 것도 나쁘진 않을 듯 싶다.
collection_data_inner = pd.merge(meal_demand_df_sum.reset_index(), menu_df_sum.reset_index(),
                                 on=['일자', '식사명'], how='inner').set_index(['일자', '식사명'])
collection_data_inner = pd.merge(collection_data_inner.reset_index(), embedded_menu_list_padded_df.reset_index(),
                                 on=['일자', '식사명'], how='inner').set_index(['일자', '식사명'])
collection_data_inner.to_csv("collection_data_inner.csv", encoding='utf-8')


'''

# new_menu_list = numpy.asarray(new_menu_list)

# new_menu_list_df = pandas.DataFrame(data={"식사명":list(dataframe['식사명'].values),
#                                        "식사내용":list(new_menu_list)}, index=list(dataframe['일자']))
# new_menu_list_df.to_csv('menu_preprocessed.csv', encoding='utf-8')


'''

