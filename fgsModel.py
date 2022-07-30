import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


vocab = []
dataRaw = []


# clean fg data
data = pd.read_excel("helper/fgs7900.xlsx")
labels = pd.read_excel("datasets/dataset7900.xlsx")
# data = pd.read_excel("helper/fgs100.xlsx")
# labels = pd.read_excel("datasets/test100.xlsx")


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
    
    # dataY.append(labels.iloc[smile, 3])

mostFgs = len(sorted(dataX, reverse=True, key=len)[0])
buffer = 0
for smile in range(len(dataX)):
    for i in range(mostFgs - len(dataX[smile - buffer])):
        dataX[smile - buffer].append(0)  # standardize length
    
    try:
        dataX[smile - buffer].append(float(labels.iloc[smile, 5][:-3]))  # append mass
        dataY.append(labels.iloc[smile - buffer, 3])
    except (ValueError, TypeError) as e:
        del dataX[smile-buffer]
        buffer += 1
print(len(dataX))

# scale labels
minDataY = min(dataY)
maxDataY = max(dataY)
dataY = (dataY - minDataY) / maxDataY

# shuffle data
data = list(zip(dataX, dataY))
random.shuffle(data)
dataX, dataY = zip(*data)

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
print(np.shape(trainX), buffer)

model = Sequential()
model.add(Input(shape=(mostFgs+1,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=100, batch_size=4, shuffle=True)
mse, mae = model.evaluate(testX, testY, verbose=0)

output, expectedOutput = [], []
for i, prediction in enumerate(model.predict(testX)):
    # print(f"Predicted: {prediction[0]*maxDataY+minDataY} Actual: {testY[i]*maxDataY+minDataY}")
    output.append(prediction[0]*maxDataY+minDataY)
    expectedOutput.append(testY[i]*maxDataY+minDataY)

print('MSE: %.3f,  MAE: %.3f' % (mse, mae))
print('Scaled mean error: %.3f' % ((np.abs((np.array(output) - np.array(expectedOutput)))).mean()))


history_dict = history.history
acc = history_dict['mse']
val_acc = history_dict['val_mse']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training mse')
plt.plot(epochs, val_acc, 'b', label='Validation mse')
plt.title('Training and validation error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
