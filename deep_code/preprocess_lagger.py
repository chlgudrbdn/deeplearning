# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime
import os
# http://rfriend.tistory.com/264?category=675917

filename = os.getcwd() + '\dataset\menu_preprocessed.csv'
# filename = os.getcwd() + '\menu_preprocessed.csv'
df = pd.read_csv(filename, header=0)


df.fillna(method='ffill') # 위의 값으로 채우기

#
dataset = df.values
dataset = dataset.astype('float32')
dates = df[:, 1]
ts = Series(df[:, 1:], index=dates)
ts.interpolate(method='time') # 시계열 날짜 index를 기준으로 결측값 보간(선형)


