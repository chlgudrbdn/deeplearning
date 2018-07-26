# -*- coding: utf-8 -*-
from datetime import datetime
from pandas import read_csv

# load data
def parse(x):
	return datetime.strptime(x, '%Y%m%d')

dataset = read_csv('date_And_ironorePrice-forecast.csv',  parse_dates = ['X'], date_parser=parse)