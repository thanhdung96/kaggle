import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print('Prepare data')
melbourneData = pd.read_csv('../melb_data.csv').dropna()
y = melbourneData.Price
melbourneData = melbourneData.drop(['Price'], axis=1)
melbourneData = melbourneData.select_dtypes(exclude=['object'])
print(melbourneData.shape)
print('Splitting sets')
train_X, test_X, train_y, test_y = train_test_split(
    melbourneData,
    y,
    train_size=0.9,
    test_size=0.1,
    random_state=0,
    shuffle=True
)

featModel = RandomForestRegressor()
featModel.fit(train_X, train_y)
print(featModel.feature_importances_)
