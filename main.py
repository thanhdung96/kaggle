import pandas as pd

melborneData = pd.read_csv('./melb_data.csv')
print(melborneData.describe())
