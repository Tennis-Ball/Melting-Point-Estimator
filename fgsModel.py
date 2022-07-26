import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

import pandas as pd
import numpy as np


vocab = []
dataRaw = []


# clean fg data
data = pd.read_excel("helper/fgs500.xlsx")
labels = pd.read_excel("datasets/test500.xlsx")

for i in range(len(data)):
    smile = data.iloc[i, 0]
    dataRaw.append([])
    for found in data.iloc[i, 1:]:
        if isinstance(found, float):  # break if no more fgs
            break
        
        # convert string back into list
        fgInfo = found[2:-2].split("', '")
        fg = fgInfo[0]
        if len(fgInfo) == 1:
            locations = []
        else:
            locations = fgInfo[1][1:-1].split('), ')

        for location in range(len(locations)):  # find starting location
            locations[location] = locations[location][1:locations[location].find(",")]
            dataRaw[i].append([fg, int(locations[location])])
        
        vocab.append(fg)

# set vocabulary for tokenization
vocab = list(set(vocab))
vocabDict = {}
for i in range(len(vocab)):
    vocabDict[vocab[i]] = i+1
dataX = []
dataY = []

for smile in range(len(dataRaw)):
    dataX.append([])
    for fg in sorted(sorted(dataRaw[smile]), key=lambda x: (x[1], x[0])):  # sort by fg length and location
        dataX[smile].append(vocabDict[fg[0]])  # append tokenized point
    
    dataY.append(labels.iloc[smile, 3])

mostFgs = len(sorted(dataX, reverse=True, key=len)[0])
for smile in range(len(dataX)):
    for i in range(mostFgs - len(dataX[smile])):
        dataX[smile].append(0)  # standardize length

# scale labels
minDataY = min(dataY)
maxDataY = max(dataY)
dataY = (dataY - minDataY) / maxDataY

# convert to numpy arrays
trainX = np.array(dataX[:int(len(dataX)*0.6)])
trainY = np.array(dataY[:int(len(dataY)*0.6)])
valX = np.array(dataX[int(len(dataX)*0.6):int(len(dataX)*0.8)])
valY = np.array(dataY[int(len(dataY)*0.6):int(len(dataY)*0.8)])
testX = np.array(dataX[int(len(dataX)*0.8):])
testY = np.array(dataY[int(len(dataY)*0.8):])

print(len(trainX), len(trainY))
print(len(valX), len(valY))
print(len(testX), len(testY))

model = Sequential()
model.add(Input(shape=(mostFgs,)))
# model.add(Dense(512, activation='sigmoid'))
# model.add(Dense(256, activation='sigmoid'))
# model.add(Dense(256, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(trainX, trainY, validation_data=(valX, valY), epochs=100, batch_size=1, shuffle=True)
mse, mae = model.evaluate(testX, testY, verbose=0)
print('MSE: %.3f,  MAE: %.3f' % (mse, mae))

for i, prediction in enumerate(model.predict(testX)):
    print(f"Predicted: {prediction[0]*maxDataY+minDataY} Actual: {testY[i]*maxDataY+minDataY}")
