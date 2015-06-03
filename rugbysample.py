# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:58:44 2015

@author: Trevor1
"""

import pandas as pd

sample = pd.read_csv('sample.csv',sep=',')

feature_cols = ['bath', 'harlequins', 'leicester','bath_opp','harlequins_opp','leicester_opp']

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(fit_intercept=False)

x_train = sample[feature_cols]
y_train = sample.win_margin

logreg.fit(x_train, y_train)
logreg.coef_
sum(logreg.coef_)

