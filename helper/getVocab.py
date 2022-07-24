import pandas as pd


ds = pd.read_excel("dataset.xlsx")  # import dataset from excel
rawChars = []
for i in range(len(ds)):
    for c in ds.iloc[i, 2]:
        rawChars.append(c)

uniqueChars = sorted(list(set(rawChars)))
vocabDict = {}
for i in range(len(uniqueChars)):
    vocabDict[uniqueChars[i]] = i + 1

print(vocabDict)
