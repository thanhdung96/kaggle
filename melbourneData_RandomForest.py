import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

melbourneData = pd.read_csv('./melb_data.csv').dropna()
print(melbourneData.columns)
print('-----------')
y = melbourneData.Price
feats = ['Distance', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourneData[feats]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

melbourneModel = RandomForestRegressor(random_state=1)
melbourneModel.fit(train_X, train_y)
predictedPrice = melbourneModel.predict(test_X)
print(mean_absolute_error(predictedPrice, test_y))
