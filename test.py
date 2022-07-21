# # import tensorflow
import pandas as pd
from rdkit import Chem
from constants import fgSmiles

ds = pd.read_excel("test.xlsx")
fgs = []

for smileNotationIdx in range(len(ds.iloc[:, 2])):
    sampleMol = Chem.MolFromSmiles(ds.iloc[smileNotationIdx, 2])
    fgs.append([])

    for fgSmile in fgSmiles:
        try:
            fg = Chem.MolFromSmarts(fgSmile)
            foundFg = sampleMol.GetSubstructMatches(fg)
            if foundFg != () and foundFg:
                fgs[smileNotationIdx].append([str(ds.iloc[smileNotationIdx, 2]), str(fgSmile), str(foundFg)])
        except Exception as e:
            pass

print(pd.DataFrame(fgs))
pd.DataFrame(fgs).to_excel("output.xlsx")
