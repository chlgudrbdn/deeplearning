#-*- coding: utf-8 -*-
import pandas
import numpy
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, sys
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict, defaultdict
from keras.preprocessing import sequence

filename = os.getcwd() + '\menu.csv'
# filename = os.getcwd() + '\dataset\menu.csv'
dataframe = pandas.read_csv(filename)
dataset = dataframe.values
# dataset = dataset.astype('float32')
# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
menu = dataset[:, 3]
menu_dictionary = set(['깍두기(손칼)'])
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

