#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:39:32 2020

@author: sylvia
"""

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
ds = read_csv('./data/lstm/airline-passengers.csv', usecols=[1], engine='python')


def create_ts_set(d, lb=1):
    dataX, dataY = [], []
    for i in range(len(d) - lb - 1):
        a = d[i:(i + lb), 0]
        dataX.append(a)
        dataY.append(d[i + lb, 0])
    return numpy.array(dataX), numpy.array(dataY)


def create_win_dataset(d, lb=1):
    dataX, dataY = [], []
    for i in range(len(d) - lb - 1):
        a = d[i:(i + lb), 0]
        dataX.append(a)
        dataY.append(d[i + lb, 0])
    return numpy.array(dataX), numpy.array(dataY)


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(ds)

train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

look_back = 3
trainX, trainY = create_ts_set(train, look_back)
testX, testY = create_win_dataset(test, look_back)

# reshape input to be [samples, time steps, features]

winTrainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

trainX, trainY = create_ts_set(train, look_back)
testX, testY = create_ts_set(test, look_back)

tsTrainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))


print("Windows Training Data:", winTrainX)
print("Time Steps Training Data:", tsTrainX)

print(trainX.shape[1])