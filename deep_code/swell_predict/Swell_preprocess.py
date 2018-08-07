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

filename = os.getcwd() + '\swell_for_preprocess.csv'
# filename = os.getcwd() + '\dataset\swell_for_preprocess.csv'
df = pd.read_csv(filename)
dataset = df.values
# dataset = dataset.astype('float32')
# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
WallPo_swell = pd.read_csv('(과제2) 관련. 월포 파고부이 데이터(15~17).csv')
WallPo_swell = pd.read_csv('(과제2) 관련. 월포 파고부이 데이터(15~17).csv')

zeroMatrix = np.zeros(((365*4)+1, 24))  # +1은 2016년 윤달. 7시부터 다음날 7시까지 24칸. 2014년 부터 2017년 자료를 대상.
# 일단 오버샘플링도 고려할 필요는 있다만 그건 나중에 생각해보자.
df['좌']
df['우']

# 30분 짜리도 있긴 하지만 딱히 정밀도에 영향을 주거나 할 것 같진 않다.


zeroDF = pd.DataFrame(data=zeroMatrix)
zeroDF.columns = ["07:00~08:00", "08:00~09:00", "09:00~10:00", "10:00~11:00", "11:00~12:00", "12:00~13:00",
                  "13:00~14:00", "14:00~15:00",	"15:00~16:00", "16:00~17:00", "17:00~18:00", "18:00~~19:00",
                  "19:00~20:00", "20:00~21:00", "21:00~22:00", "22:00~23:00", "23:00~24:00", "00:00~01:00",
                  "01:00~02:00", "02:00~03:00", "03:00~04:00", "04:00~05:00", "05:00~06:00", "06:00~07:00"]
zeroDF

for menus in menu:
    menu_elem = menus.split(",")
    menu_dictionary = menu_dictionary.union(set(menu_elem))  # 메뉴 총 개수 1878. 한끼에 최대 메뉴 19

MenuList = list(menu_dictionary)
menu_dict = dict(list(zip(MenuList, range(len(MenuList)))))
new_menu_list = []
MaxLength = 0
for menus in menu:
    menu_elem = menus.split(",")
    if len(menu_elem) >= MaxLength:
        MaxLength = len(menu_elem)
        # print(menu_elem)
    new_menu_elem = []
    for menuItem in menu_elem:
        new_menu_elem.append(menu_dict[menuItem])
    new_menu_list.append(new_menu_elem)
print(MaxLength)
# new_menu_list = numpy.asarray(new_menu_list)
new_menu_list = sequence.pad_sequences(new_menu_list, maxlen=MaxLength)
new_menu_list = pandas.DataFrame(data=new_menu_list)
new_menu_list.to_csv('menu_preprocessed.csv', encoding='utf-8')

# dataframe.merge(new_menu_list, )

