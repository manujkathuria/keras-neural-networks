#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:48:34 2020

@author: sylvia
"""

# Time Series Anomaly Detection

# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load dataset
df = pd.read_csv('D:/Workspace/pycharm/keras-ann/data/ae/real_3.csv', parse_dates=['timestamp'], index_col='timestamp')

print(df.head(6))

# Plot data
plt.plot(df, label='time_Series data')
plt.legend()
plt.show()

# Preprocessing

train_size = int(len(df) * 0.95)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)
print(train)

# Rescale data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train[['value']])

train['value'] = scaler.transform(train[['value']])
test['value'] = scaler.transform(test[['value']])


# we’ll split the data into subsequences

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# Create sequences 30 days worth of historical data
TIME_STEPS = 10

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(
    train[['value']],
    train.value,
    TIME_STEPS
)

for i in range(len(X_train)):
    print(X_train[i], y_train[i])

X_test, y_test = create_dataset(
    test[['value']],
    test.value,
    TIME_STEPS
)
print(len(X_test))

print(X_train.shape)

X_train.shape[1]
# 10 timestep
X_train.shape[2]
# 1 features


# Make a model - LSTM Autoencoder

# Encoder
model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))

# Decoder
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
model.compile(loss='mae', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=12,
    validation_split=0.1,
    shuffle=False
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()

# Predictions on Train

X_train_pred = model.predict(X_train)
len(X_train_pred)

print(X_train_pred[1])
print(X_train[1])

# mae - mean absolute error b/w actual and predicted

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
train_mae_loss = train_mae_loss.flatten()
len(train_mae_loss)  # 1377

# Draw distribution plot for loss/cost function / error
import seaborn as sns

sns.distplot(train_mae_loss, bins=50, kde=True)
plt.show()


def statistical_threshold(loss_function):
    import statistics

    m = statistics.mean(loss_function)
    # 3 standard deviation as outlier
    print("Mean:", m)
    sd = statistics.stdev(loss_function, xbar=m)
    print("SD:", sd)
    new_std_dev = sd * 3
    print("New SD:", new_std_dev)
    return new_std_dev


THRESHOLD = statistical_threshold(train_mae_loss)

print(THRESHOLD)

# Predictions on test

X_test_pred = model.predict(X_test)
len(X_test_pred)

X_test_pred

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
test_mae_loss

import seaborn as sns

sns.distplot(test_mae_loss, bins=50, kde=True)

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['anomaly'] = test_score_df['anomaly'].astype(int)
test_score_df['value'] = test[TIME_STEPS:].value
print(test_score_df)
len(test_score_df)
test_predict = test_score_df['anomaly']
test_predict

plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend()
plt.show()

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()
print(len(anomalies))
print(anomalies)

# Compare with ground truth using confusion matrix

df1 = pd.read_csv('D:/Workspace/pycharm/keras-ann/data/ae/real_3.csv', parse_dates=['timestamp'], index_col='timestamp')
df1 = df1.drop(['value'], axis=1)
test_actual = df1.iloc[1397:]

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(test_actual)
print(test_predict)
results = confusion_matrix(test_actual, test_predict)

# Printing all results compared with ground truth
print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(test_actual, test_predict))
print('Report : ')
print(classification_report(test_actual, test_predict))

# Plot anomalies
plt.plot(
    test[TIME_STEPS:].index,
    scaler.inverse_transform(test[TIME_STEPS:].value),
    label='value'
)
plt.show()

sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.value),
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
)

plt.xticks(rotation=25)
plt.legend()
plt.show()
