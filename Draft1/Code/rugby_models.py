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

## turning team and opponent into dumby variables
## now my model will "know" which teams are playing

for elem in rugby['team'].unique():
    rugby[str(elem) + '_home'] = rugby['team'] == elem

for elem in rugby['opponent'].unique():
    rugby[str(elem) + '_opp'] = rugby['opponent'] == elem

teams = (rugby['team'].unique() + '_home')
opps = (rugby['team'].unique() + '_opp')

## transforming booleans into 1's, 0's

rugby[teams] = rugby[teams].astype(int)
rugby[opps] = rugby[opps].astype(int)


## naming feature columns

feature_cols = ['home','harlequins_home', 'exeter_home', 'sale_home', 'wasps_home',
       'bath_home', 'gloucester_home', 'irish_home', 'leicester_home',
       'worcester_home', 'saracens_home', 'newcastle_home',
       'northampton_home', 'welsh_home','harlequins_opp', 'exeter_opp', 'sale_opp', 'wasps_opp', 'bath_opp',
       'gloucester_opp', 'irish_opp', 'leicester_opp', 'worcester_opp',
       'saracens_opp', 'newcastle_opp', 'northampton_opp', 'welsh_opp']

## training on the first 18 rounds of each season

x_train = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_train = rugby[rugby.round < 19][rugby.round > 1].points_diff

## testing on 19, 20, 21, and 22 rounds

x_test = rugby[rugby.round > 18][feature_cols]
y_test = rugby[rugby.round > 18].points_diff

## fit the model

logreg.fit(x_train, y_train)

## create predictions

y_pred = logreg.predict(x_test)

## test accuracy

metrics.accuracy_score(y_test,y_pred)

logreg.coef_

x_list = list(range(0,len(y_pred)))

false_positive = []
false_negative = []

for x in x_list:
    list_y = list(y_test)
    list_x = list(x_test.index)
    if y_pred[x] > list_y[x]:
        false_positive.append(list_x[x])
    elif y_pred[x] < list_y[x]:
        false_negative.append(list_x[x])

###################################################################
###################################################################
###################################################################

feature_cols = ['wins','wins_opp']

x_train = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_train = rugby[rugby.round < 19][rugby.round > 1].win

x_test = rugby[rugby.round > 18][feature_cols]
y_test = rugby[rugby.round > 18].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

metrics.accuracy_score(y_test,y_pred)

logreg.coef_

###################################################################
###################################################################
###################################################################

feature_cols = ['home']

x_train = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_train = rugby[rugby.round < 19][rugby.round > 1].win

x_test = rugby[rugby.round > 18][feature_cols]
y_test = rugby[rugby.round > 18].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

metrics.accuracy_score(y_test,y_pred)

logreg.coef_

###################################################################
###################################################################
###################################################################

feature_cols = ['home','wins','wins_opp']

x_train = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_train = rugby[rugby.round < 19][rugby.round > 1].win

x_test = rugby[rugby.round > 18][feature_cols]
y_test = rugby[rugby.round > 18].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

metrics.accuracy_score(y_test,y_pred)

logreg.coef_

###################################################################
###################################################################
###################################################################



###################################################################
###################################################################
###################################################################

feature_cols = ['home','win_percent','opp_win_percent']

knn = KNeighborsClassifier(n_neighbors=15)   

x_trk = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_trk = rugby[rugby.round < 19][rugby.round > 1].win

x_tk = rugby[rugby.round > 18][feature_cols]
y_tk = rugby[rugby.round > 18].win

knn.fit(x_trk, y_trk)
y_pk = knn.predict(x_tk)

metrics.accuracy_score(y_tk, y_pk)

k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_trk, y_trk)
    y_pk = knn.predict(x_tk)
    scores.append(metrics.accuracy_score(y_tk, y_pk))

scores