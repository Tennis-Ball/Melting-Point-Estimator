import requests
import pandas as pd


ds = pd.read_excel("datasets/dataset.xlsx")
masses = []

for i in range(2500):
    try:
        r = requests.get("http://www.chemspider.com/Chemical-Structure." + str(ds.iloc[i, 4]) + ".html")
        mass = r.text[r.text.find("Average mass") + 19:]
        mass = mass[:mass.find("<")]
    except Exception as e:
        print(e)
        masses.append(e)
        continue

    print(mass)
    masses.append(mass)

print(masses)
pd.DataFrame(masses).to_excel("masses2500.xlsx")
