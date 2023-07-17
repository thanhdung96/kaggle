import pandas as pd

gameData = pd.read_csv('./backloggd_games.csv')
gameData.Plays = (gameData.Plays.replace(r'[KM]+$', '', regex=True).astype(float) *
                  gameData.Plays.str.extract(r'[\d\.]+([KM]+)', expand=False)
                        .fillna(1)
                        .replace(['K','M'], [10**3, 10**6]).astype(int)
                  )
