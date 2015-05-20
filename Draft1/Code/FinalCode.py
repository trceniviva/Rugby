# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:52:35 2015

@author: Trevor1
"""

### Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
from sklearn.neighbors import KNeighborsClassifier  # import class 

### Bringing in the dataset

rugby = pd.read_table('rugbyfinal.csv', sep=',')

## creating last two features

rugby['win_percent'] = (rugby['wins'] / (rugby['round'] - 1))
rugby['opp_win_percent'] = (rugby['wins_opp'] / (rugby['round'] - 1))

rugby['win_percent'] = rugby['win_percent'].fillna(0)
rugby['opp_win_percent'] = rugby['opp_win_percent'].fillna(0)

## turning team and opponent into dumby variables
## now my model will "know" which teams are playing

for elem in rugby['team'].unique():
    rugby[str(elem) + '_home'] = rugby['team'] == elem

for elem in rugby['team'].unique():
    rugby[str(elem)] = rugby['team'] == elem

for elem in rugby['opponent'].unique():
    rugby[str(elem) + '_opp'] = rugby['opponent'] == elem

teams = rugby['team'].unique()
teams_home = (rugby['team'].unique() + '_home')
opps = (rugby['team'].unique() + '_opp')

## transforming booleans into 1's, 0's

rugby[teams_home] = rugby[teams_home].astype(int)
rugby[opps] = rugby[opps].astype(int)
rugby[teams] = rugby[teams].astype(int)

rugby['harlequins'] = rugby['harlequins'] + rugby['harlequins_opp']
del rugby['harlequins_opp']
rugby['leicester'] = rugby['leicester'] + rugby['leicester_opp']
del rugby['leicester_opp']
rugby['exeter'] = rugby['exeter'] + rugby['exeter_opp']
del rugby['exeter_opp']
rugby['sale'] = rugby['sale'] + rugby['sale_opp']
del rugby['sale_opp']
rugby['wasps'] = rugby['wasps'] + rugby['wasps_opp']
del rugby['wasps_opp']
rugby['bath'] = rugby['bath'] + rugby['bath_opp']
del rugby['bath_opp']
rugby['gloucester'] = rugby['gloucester'] + rugby['gloucester_opp']
del rugby['gloucester_opp']
rugby['irish'] = rugby['irish'] + rugby['irish_opp']
del rugby['irish_opp']
rugby['worcester'] = rugby['worcester'] + rugby['worcester_opp']
del rugby['worcester_opp']
rugby['saracens'] = rugby['saracens'] + rugby['saracens_opp']
del rugby['saracens_opp']
rugby['newcastle'] = rugby['newcastle'] + rugby['newcastle_opp']
del rugby['newcastle_opp']
rugby['northampton'] = rugby['northampton'] + rugby['northampton_opp']
del rugby['northampton_opp']
rugby['welsh'] = rugby['welsh'] + rugby['welsh_opp']
del rugby['welsh_opp']

## creating dummy variables for seasons

for elem in rugby['season'].unique():
    rugby['season_' + str(elem)] = rugby['season'] == elem

seasons = ('season_' + rugby['season'].unique())

rugby[seasons] = rugby[seasons].astype(int)



## naming feature columns

feature_cols = ['harlequins_home', 'exeter_home', 'sale_home', 'wasps_home',
       'bath_home', 'gloucester_home', 'irish_home', 'leicester_home',
       'worcester_home', 'saracens_home', 'newcastle_home',
       'northampton_home', 'welsh_home','harlequins', 'exeter', 'sale', 'wasps', 'bath', 'gloucester',
       'irish', 'leicester', 'worcester', 'saracens', 'newcastle',
       'northampton', 'welsh']

## training on the first 18 rounds of each season

x_train = rugby[rugby.season != '14_15'][rugby.round < 18][feature_cols]
y_train = rugby[rugby.season != '14_15'][rugby.round < 18].win

## testing on 19, 20, 21, and 22 rounds

x_test = rugby[rugby.season != '14_15'][rugby.round > 19][feature_cols]
y_test = rugby[rugby.season != '14_15'][rugby.round > 19].win

## fit the model

logreg.fit(x_train, y_train)

## create predictions

y_pred = logreg.predict(x_test)

y_pred_final = logreg.predict(x_test)

## test accuracy

metrics.accuracy_score(y_test,y_pred)

logreg.coef_

######################### 

x_train_final = rugby[rugby.season != '14_15'][feature_cols]
y_train_final = rugby[rugby.season != '14_15'].win

x_test_final = rugby[rugby.season == '14_15'][feature_cols]
y_test_final = rugby[rugby.season == '14_15'].win

logreg.fit(x_train_final, y_train_final)

y_pred_final = logreg.predict(x_test_final)

metrics.accuracy_score(y_test_final,y_pred_final)

######################### 