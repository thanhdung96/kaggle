import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

print('Prepare data')
melbourneData = pd.read_csv('../melb_data.csv').dropna()
y = melbourneData.Price
melbourneData = melbourneData.drop(['Price'], axis=1)
melbourneData = melbourneData.select_dtypes(exclude=['object'])
print('Splitting sets')
train_X, test_X, train_y, test_y = train_test_split(
    melbourneData,
    y,
    train_size=0.9,
    test_size=0.1,
    random_state=0,
    shuffle=True
)

print('Selecting features')
featModel = RandomForestRegressor(random_state=0)
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
pipeline = Pipeline(steps=[('s', rfe), ('m', featModel)])
pipeline.fit(train_X, train_y)
print(rfe.get_support())
print('----------')
print(rfe.ranking_)
print('----------')
print(melbourneData.columns)

# print('Fitting model')
# melbourneModel = RandomForestRegressor(random_state=0)
# melbourneModel.fit(train_X, train_y)
# print('Predict...')
# predictedPrice = melbourneModel.predict(test_X)
# print(mean_absolute_error(predictedPrice, test_y))
