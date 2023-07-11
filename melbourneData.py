import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourneData = pd.read_csv('./melb_data.csv').dropna()
print(melbourneData.columns)
print('-----------')
y = melbourneData.Price
feats = ['Rooms', 'Bathroom', 'Bedroom2', 'Landsize', 'BuildingArea', 'YearBuilt']
X = melbourneData[feats]

melbourneModel = DecisionTreeRegressor(random_state=1)
melbourneModel.fit(X, y)
print(y.head())
print('-----------')
print(melbourneModel.predict(X.head()))
