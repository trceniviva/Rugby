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

full_cols = []

feature_cols = ['home','avgPD','avgPD_opp']

x_tr18 = rugby[rugby.round < 18][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 18][rugby.round > 1].win

x_t18 = rugby[rugby.round == 18][feature_cols]
y_t18 = rugby[rugby.round == 18].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

full_cols.append(metrics.accuracy_score(y_t18,y_p18))

###################################################################
###################################################################
###################################################################

x_tr19 = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_tr19 = rugby[rugby.round < 19][rugby.round > 1].win

x_t19 = rugby[rugby.round == 19][feature_cols]
y_t19 = rugby[rugby.round == 19].win

logreg.fit(x_tr19, y_tr19)

y_p19 = logreg.predict(x_t19)

full_cols.append(metrics.accuracy_score(y_t19,y_p19))

###################################################################
###################################################################
###################################################################

x_tr20 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr20 = rugby[rugby.round < 20][rugby.round > 1].win

x_t20 = rugby[rugby.round == 20][feature_cols]
y_t20 = rugby[rugby.round == 20].win

logreg.fit(x_tr20, y_tr20)

y_p20 = logreg.predict(x_t20)

full_cols.append(metrics.accuracy_score(y_t20,y_p20))

###################################################################
###################################################################
###################################################################

x_tr21 = rugby[rugby.round < 21][rugby.round > 1][feature_cols]
y_tr21 = rugby[rugby.round < 21][rugby.round > 1].win

x_t21 = rugby[rugby.round == 21][feature_cols]
y_t21 = rugby[rugby.round == 21].win

logreg.fit(x_tr21, y_tr21)

y_p21 = logreg.predict(x_t21)

full_cols.append(metrics.accuracy_score(y_t21,y_p21))

###################################################################
###################################################################
###################################################################

just_wins = []

feature_cols = ['wins','wins_opp']

x_tr18 = rugby[rugby.round < 18][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 18][rugby.round > 1].win

x_t18 = rugby[rugby.round == 18][feature_cols]
y_t18 = rugby[rugby.round == 18].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

just_wins.append(metrics.accuracy_score(y_t18,y_p18))

###################################################################
###################################################################
###################################################################

x_tr19 = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_tr19 = rugby[rugby.round < 19][rugby.round > 1].win

x_t19 = rugby[rugby.round == 19][feature_cols]
y_t19 = rugby[rugby.round == 19].win

logreg.fit(x_tr19, y_tr19)

y_p19 = logreg.predict(x_t19)

just_wins.append(metrics.accuracy_score(y_t19,y_p19))

###################################################################
###################################################################
###################################################################

x_tr20 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr20 = rugby[rugby.round < 20][rugby.round > 1].win

x_t20 = rugby[rugby.round == 20][feature_cols]
y_t20 = rugby[rugby.round == 20].win

logreg.fit(x_tr20, y_tr20)

y_p20 = logreg.predict(x_t20)

just_wins.append(metrics.accuracy_score(y_t20,y_p20))

###################################################################
###################################################################
###################################################################

x_tr21 = rugby[rugby.round < 21][rugby.round > 1][feature_cols]
y_tr21 = rugby[rugby.round < 21][rugby.round > 1].win

x_t21 = rugby[rugby.round == 21][feature_cols]
y_t21 = rugby[rugby.round == 21].win

logreg.fit(x_tr21, y_tr21)

y_p21 = logreg.predict(x_t21)

just_wins.append(metrics.accuracy_score(y_t21,y_p21))

###################################################################
###################################################################
###################################################################

justpds = []

feature_cols = ['avgPD','avgPD_opp']

x_tr18 = rugby[rugby.round < 18][rugby.round > 1][feature_cols]
y_tr18 = rugby[rugby.round < 18][rugby.round > 1].win

x_t18 = rugby[rugby.round == 18][feature_cols]
y_t18 = rugby[rugby.round == 18].win

logreg.fit(x_tr18, y_tr18)

y_p18 = logreg.predict(x_t18)

justpds.append(metrics.accuracy_score(y_t18,y_p18))

###################################################################
###################################################################
###################################################################

x_tr19 = rugby[rugby.round < 19][rugby.round > 1][feature_cols]
y_tr19 = rugby[rugby.round < 19][rugby.round > 1].win

x_t19 = rugby[rugby.round == 19][feature_cols]
y_t19 = rugby[rugby.round == 19].win

logreg.fit(x_tr19, y_tr19)

y_p19 = logreg.predict(x_t19)

justpds.append(metrics.accuracy_score(y_t19,y_p19))

###################################################################
###################################################################
###################################################################

x_tr20 = rugby[rugby.round < 20][rugby.round > 1][feature_cols]
y_tr20 = rugby[rugby.round < 20][rugby.round > 1].win

x_t20 = rugby[rugby.round == 20][feature_cols]
y_t20 = rugby[rugby.round == 20].win

logreg.fit(x_tr20, y_tr20)

y_p20 = logreg.predict(x_t20)

justpds.append(metrics.accuracy_score(y_t20,y_p20))

###################################################################
###################################################################
###################################################################

x_tr21 = rugby[rugby.round < 21][rugby.round > 1][feature_cols]
y_tr21 = rugby[rugby.round < 21][rugby.round > 1].win

x_t21 = rugby[rugby.round == 21][feature_cols]
y_t21 = rugby[rugby.round == 21].win

logreg.fit(x_tr21, y_tr21)

y_p21 = logreg.predict(x_t21)

justpds.append(metrics.accuracy_score(y_t21,y_p21))


sum(full_cols)/4
sum(just_wins)/4
sum(justpds)/4

