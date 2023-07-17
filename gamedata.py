import pandas as pd
from ast import literal_eval
from pandas.core.series import Series
from sklearn.model_selection import train_test_split


def calculateKilo(col: Series, naReplacemane: int = 0) -> Series:
    newcol = (col.replace(r'[KM]+$', '', regex=True).astype(float) *
              col.str.extract(r'[\d\.]+([KM]+)', expand=False)
              .fillna(naReplacemane)
              .replace(['K', 'M'], [10 ** 3, 10 ** 6]).astype(int)
              )

    return newcol


def parseList(col: Series) -> Series:
    newcol = col.apply(literal_eval)

    return newcol


gameData = pd.read_csv('./backloggd_games.csv')

# preprocess abbreviated numeric values
gameData.Plays = calculateKilo(gameData.Plays, 0)
gameData.Playing = calculateKilo(gameData.Playing, 0)
gameData.Backlogs = calculateKilo(gameData.Backlogs, 0)
gameData.Wishlist = calculateKilo(gameData.Wishlist, 0)
gameData.Lists = calculateKilo(gameData.Lists, 0)
gameData.Reviews = calculateKilo(gameData.Reviews, 0)

# preprocess array-formed string to array of strings
gameData.Developers = parseList(gameData.Developers)
gameData.Platforms = parseList(gameData.Platforms)
gameData.Genres = parseList(gameData.Genres)

print(gameData.Developers[0][0])
