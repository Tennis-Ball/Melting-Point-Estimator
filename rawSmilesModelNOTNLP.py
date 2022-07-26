import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

import pandas as pd
import numpy as np


ds = pd.read_excel("datasets/test500.xlsx")  # import dataset from excel
vocabDict = {'#': 1, '%': 2, '(': 3, ')': 4, '+': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, '=': 19, '@': 20, 'A': 21, 'B': 22, 'C': 23, 'F': 24, 'G': 25, 'H': 26, 'I': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'S': 32, 'T': 33, 'Z': 34, '[': 35, '\\': 36, ']': 37, 'a': 38, 'b': 39, 'c': 40, 'e': 41, 'g': 42, 'i': 43, 'l': 44, 'n': 45, 'o': 46, 'p': 47, 'r': 48, 's': 49}
longestSmile = max(list(len(s) for s in ds.iloc[:, 2]))

# initialize data/label sets
trainX = []
trainY = []
valX = []
valY = []
testX = []
testY = []

for i in range(len(ds)):
    convertedData = []
    for c in range(longestSmile):
            if c >= len(ds.iloc[i, 2]):
                convertedData.append(0)
            else:
                convertedData.append(vocabDict[ds.iloc[i, 2][c]])
    convertedData.append(float(ds.iloc[i, 5][:-3]))  # append numeric mass

    if i < len(ds) // (5/3):  # append first 60% of data to training
        trainX.append(convertedData)
        trainY.append(ds.iloc[i, 3])

    elif i < len(ds) // (5/4):  # append next 20% of data to validation
        valX.append(convertedData)
        valY.append(ds.iloc[i, 3])

    else:  # append last 20% of data to testing
        testX.append(convertedData)
        testY.append(ds.iloc[i, 3])

# standardize melting points to range from 0 to 1
minTrainY = min(trainY)
maxTrainY = max(trainY)
trainY = (trainY - minTrainY) / maxTrainY
valY = (valY - min(valY)) / max(valY)
originalTestY = testY
testY = (testY - min(testY)) / max(testY)

# convert to numpy arrays
trainX = np.array(trainX)
trainY = np.array(trainY)
valX = np.array(valX)
valY = np.array(valY)
testX = np.array(testX)
testY = np.array(testY)

model = Sequential()
model.add(Input(shape=(longestSmile+1,)))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
# model.add(Dense(256, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(trainX, trainY, validation_data=(valX, valY), epochs=500, batch_size=4, shuffle=True)
mse, mae = model.evaluate(testX, testY, verbose=0)
print('MSE: %.3f,  MAE: %.3f' % (mse, mae))

for i, prediction in enumerate(model.predict(testX)):
    print(f"Predicted: {prediction[0]*maxTrainY+minTrainY} Actual: {originalTestY[i]}")
