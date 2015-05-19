# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:52:35 2015

@author: Trevor1
"""

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

rugby = pd.read_table('rugbyfinal.csv', sep=',')

del rugby['game']

rugby['win_percent'] = (rugby['wins'] / (rugby['round'] - 1))
rugby['opp_win_percent'] = (rugby['wins_opp'] / (rugby['round'] - 1))

feature_cols = ['round','home','avgPD','avgPD_opp','wins','wins_opp']

full_cols = []

x_train = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_train = rugby[rugby.round < 20][rugby.round > 1].points_diff

x_test = rugby[rugby.round > 19][feature_cols]
y_test = rugby[rugby.round > 19].points_diff

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

full_cols.append(metrics.accuracy_score(y_test,y_pred))

full_cols

logreg.coef_

feature_cols = ['home','win_percent','opp_win_percent','avgPD']

full_cols = []

x_train = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_train = rugby[rugby.round < 20][rugby.round > 1].win

x_test = rugby[rugby.round > 19][feature_cols]
y_test = rugby[rugby.round > 19].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

full_cols.append(metrics.accuracy_score(y_test,y_pred))

full_cols

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

win_pred = []

feature_cols = ['wins','wins_opp']

x_tr18 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 20][rugby.round > 1].win

x_t18 = rugby[rugby.round > 19][feature_cols]
y_t18 = rugby[rugby.round > 19].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

win_pred.append(metrics.accuracy_score(y_t18,y_p18))
metrics.confusion_matrix(y_t18,y_p18)

win_pred

###################################################################
###################################################################
###################################################################

home_pred = []

feature_cols = ['home']

x_tr18 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 20][rugby.round > 1].win

x_t18 = rugby[rugby.round > 19][feature_cols]
y_t18 = rugby[rugby.round > 19].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

home_pred.append(metrics.accuracy_score(y_t18,y_p18))
metrics.confusion_matrix(y_t18,y_p18)

home_pred

###################################################################
###################################################################
###################################################################

home_win = []

feature_cols = ['home','wins','wins_opp']

x_tr18 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 20][rugby.round > 1].win

x_t18 = rugby[rugby.round > 19][feature_cols]
y_t18 = rugby[rugby.round > 19].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

home_win.append(metrics.accuracy_score(y_t18,y_p18))
metrics.confusion_matrix(y_t18,y_p18)

home_win

###################################################################
###################################################################
###################################################################

full_cols
win_pred
home_pred

###################################################################
###################################################################
###################################################################

feature_cols = ['home','avgPD','avgPD_opp','wins','wins_opp']

knn = KNeighborsClassifier(n_neighbors=10)   

x_trk = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_trk = rugby[rugby.round < 20][rugby.round > 1].win

x_tk = rugby[rugby.round > 19][feature_cols]
y_tk = rugby[rugby.round > 19].win

knn.fit(x_trk, y_trk)
y_pk = knn.predict(x_tk)

print metrics.accuracy_score(y_tk, y_pk)

k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_trk, y_trk)
    y_pk = knn.predict(x_tk)
    scores.append(metrics.accuracy_score(y_tk, y_pk))

scores