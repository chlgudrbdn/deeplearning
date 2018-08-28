#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

import time
start_time = time.time()

forecast_with_Gu = pd.read_csv('only_flatten_lstm_Gu_stateless.pyforecast_df.csv', index_col=[0])  # 확인결과 중복있다.
forecast_with_Gu.columns = ['GuForecast']
forecast_with_Wall = pd.read_csv('only_flatten_lstm_Wall_stateless.pyforecast_df.csv', index_col=[0])  # 확인결과 중복있다.
forecast_with_Wall.columns = ['WallForecast']

forecast_df_join_outer = pd.concat([forecast_with_Gu, forecast_with_Wall], axis=1, join='outer', sort=True)  # 마지막에 Y값을 붙임. colum 명은 0이 됨.

finalForeCast = []
# finalForeCast_index = []

for index, row in forecast_df_join_outer.iterrows():
    print(row.GuForecast, " : ", row.WallForecast)
    if row.GuForecast == row.WallForecast:  # 둘다 같은 값이 나온 경우
        if row.GuForecast == "":
            print("alert!")
            break
        print("same result!")
        finalForeCast.append(int(row.GuForecast))
        continue

    if math.isnan(row.WallForecast):  # 둘중 하나가 빈 경우
        print("blank one!")
        finalForeCast.append(int(row.GuForecast))
        continue
    if math.isnan(row.GuForecast):
        print("blank one!")
        finalForeCast.append(int(row.WallForecast))
        continue

    if (row.WallForecast == 1) or (row.GuForecast == 1):  # 둘다 값이 있는데 하나만 1이 나온 경우
        print("one is one!")
        finalForeCast.append(1)
        continue

time_grid = ["07:00", "08:00", "09:00", "10:00", "11:00", "12:00",
             "13:00", "14:00", "15:00", "16:00", "17:00", "18:00",
             "19:00", "20:00", "21:00", "22:00", "23:00", "00:00",
             "01:00", "02:00", "03:00", "04:00", "05:00", "06:00"]  # 시작시간 기준으로 일단 잡아본다.
test_dates = pd.read_csv('test_form.csv', usecols=[0], skiprows=[0, 1])
# test_dates = pd.read_csv(os.getcwd() +'\\deep_code\\dataset\\test_form.csv', usecols=[0], skiprows=[0, 1])
test_dates = test_dates.values.flatten().tolist()
# test_dates = set(test_dates)  # 25일 예측해야함.
test_dates.sort()

finalForeCast = np.asarray(finalForeCast)
finalForeCast = finalForeCast.reshape((-1, 24))
finalForeCast_df = pd.DataFrame(data=finalForeCast, columns=time_grid, index=test_dates)
finalForeCast_df.to_csv('swell forecast final result.csv', encoding='utf-8')