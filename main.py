import pandas as pd
from sklearn.tree import DecisionTreeRegressor

homeData = pd.read_csv('./train.csv')
# print(homeData)
# print('-----------')
y = homeData.SalePrice
feats = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = homeData[feats]

homeModel = DecisionTreeRegressor(random_state=1)
homeModel.fit(X, y)
print(y.head())
print(homeModel.predict(X.head()))
