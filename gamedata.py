import pandas as pd
from ast import literal_eval
from pandas.core.series import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def calculateKilo(col: Series, naReplacemane: int = 0) -> Series:
    newcol = (col.replace(r'[KM]+$', '', regex=True).astype(float) *
              col.str.extract(r'[\d\.]+([KM]+)', expand=False)
              .fillna(naReplacemane)
              .replace(['K', 'M'], [10 ** 3, 10 ** 6]).astype(int)
              )

    return newcol


def parseList(col: Series, type: str) -> Series:
    newcol = col.apply(literal_eval)

    return newcol.apply(tuple).astype(type)

print('Reading file')
gameData = pd.read_csv('./backloggd_games.csv')
gameData = gameData.drop(['Unnamed: 0', 'Summary'], axis=1)

print('Parsing data')
# preprocess abbreviated numeric values
gameData.Plays = calculateKilo(gameData.Plays, 0)
gameData.Playing = calculateKilo(gameData.Playing, 0)
gameData.Backlogs = calculateKilo(gameData.Backlogs, 0)
gameData.Wishlist = calculateKilo(gameData.Wishlist, 0)
gameData.Lists = calculateKilo(gameData.Lists, 0)
gameData.Reviews = calculateKilo(gameData.Reviews, 0)

# preprocess array-formed string to array of strings
gameData.Developers = parseList(gameData.Developers, 'category')
gameData.Platforms = parseList(gameData.Platforms, 'category')
gameData.Genres = parseList(gameData.Genres, 'category')

# preprocess release date
gameData.Release_Date = pd.to_datetime(gameData.Release_Date, format='%b %d, %Y')
print(gameData.Release_Date)

# encoding data
labelEncoder = LabelEncoder()
gameData.Title = labelEncoder.fit_transform(gameData.Title)

# print(gameData.columns)
# print('---------')
# y = gameData.Rating
#
# train_X, test_X, train_y, test_y = train_test_split(
#     gameData,
#     y,
#     train_size=0.9,
#     test_size=0.1,
#     random_state=0,
#     shuffle=True
# )
#
# featModel = RandomForestRegressor()
# featModel.fit(train_X, train_y)
# print(featModel.feature_importances_)
