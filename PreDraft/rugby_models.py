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

rugby = pd.read_table('rugbyfinal.csv', sep=',')

del rugby['game']

feature_cols = ['home','avgPD','avgPD_opp','wins','wins_opp']

full_cols = []

x_tr18 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 20][rugby.round > 1].win

x_t18 = rugby[rugby.round > 19][feature_cols]
y_t18 = rugby[rugby.round > 19].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

full_cols.append(metrics.accuracy_score(y_t18,y_p18))

full_cols

###################################################################
###################################################################
###################################################################

just_wins = []

feature_cols = ['home','wins','wins_opp']

x_tr18 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 20][rugby.round > 1].win

x_t18 = rugby[rugby.round > 19][feature_cols]
y_t18 = rugby[rugby.round > 19].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

just_wins.append(metrics.accuracy_score(y_t18,y_p18))

just_wins

###################################################################
###################################################################
###################################################################

predict = []

feature_cols = ['home']

x_tr18 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 20][rugby.round > 1].win

x_t18 = rugby[rugby.round > 1][feature_cols]
y_t18 = rugby[rugby.round > 1].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

predict.append(metrics.accuracy_score(y_t18,y_p18))

predict

###################################################################
###################################################################
###################################################################

full_cols
just_wins
justhome

