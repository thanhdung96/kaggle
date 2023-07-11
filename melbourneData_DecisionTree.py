import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

melbourneData = pd.read_csv('./melb_data.csv').dropna()
print(melbourneData.columns)
print('-----------')
y = melbourneData.Price
feats = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourneData[feats]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

# smallestMean = 1e10
# smallestNode = 0
# for maxNode in range(5, 1000):
#     melbourneModel = DecisionTreeRegressor(max_leaf_nodes=maxNode, random_state=0)
#     melbourneModel.fit(train_X, train_y)
#     predictedPrice = melbourneModel.predict(test_X)
#     meanError = mean_absolute_error(predictedPrice, test_y)
#     if meanError < smallestMean:
#         smallestMean = meanError
#         smallestNode = maxNode
#
# print(smallestMean)
# print(smallestNode)
# print('-----------')

# via the loop above, found that number of node is 367 is optimal
melbourneModel = DecisionTreeRegressor(max_leaf_nodes=367, random_state=0)
melbourneModel.fit(train_X, train_y)
predictedPrice = melbourneModel.predict(test_X)
print(mean_absolute_error(predictedPrice, test_y))
