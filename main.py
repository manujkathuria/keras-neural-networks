import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# dataset import
dataset = pd.read_csv('data/ann/train.csv')
dataset.head(10)
X = dataset.iloc[:, :20].values
y = dataset.iloc[:, 20:21].values

# Normalization is a technique used to change the values of an array to a common scale, without distorting
# differences in the ranges of values. It is an important step and you can check the difference in accuracies on our
# dataset by removing this step. It is mainly required in case the dataset features vary a lot as in our case the
# value of battery power is in the 1000’s and clock speed is less than 3

# Normalization
sc = StandardScaler()
X = sc.fit_transform(X)

# Next step is to one hot encode the classes. One hot encoding is a process to convert integer classes into binary
# values. Consider an example, let’s say there are 3 classes in our dataset namely 1,2 and 3. Now we cannot directly
# feed this to neural network so we convert it in the form: 1- 1 0 0 2- 0 1 0 3- 0 0 1

ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Dependencies

# Keras is a simple tool for constructing a neural network. It is a high-level framework based on tensorflow,
# theano or cntk backends.


# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

y_pred = model.predict(X_test)

# Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

# Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score

a = accuracy_score(pred, test)
print('Accuracy is:', a * 100)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
