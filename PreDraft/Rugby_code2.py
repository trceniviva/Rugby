# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:23:13 2015

@author: Trevor1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import seaborn as sns

# ( prediction - actual ) ^ 2

# calculating the predicted point difference for that team in that game
# also adding column for predicted win/loss based on the predicted point differential 

rugby = pd.read_table('aviva_fixed.csv', sep=',') # import csv sheet

predicted_points_diff = []
predicted_win_loss = []

for x in x_list:
    if x == 0:
        expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x+1])
        predicted_points_diff.append(expected)
        if expected > 0.0:
            predicted_win_loss.append(1)
        else:
            predicted_win_loss.append(0)
    elif 0 < x < 239:
        if rugby.game[x] == rugby.game[x+1]:
            expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x+1])
            predicted_points_diff.append(expected)
            if expected > 0.0:
                predicted_win_loss.append(1)
            else:
                predicted_win_loss.append(0)
        elif rugby.game[x] == rugby.game[x-1]:
            expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x-1])
            predicted_points_diff.append(expected)
            if expected > 0.0:
                predicted_win_loss.append(1)
            else:
                predicted_win_loss.append(0)
    elif x == 239:
        expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x-1])
        predicted_points_diff.append(expected)
        if expected > 0.0:
            predicted_win_loss.append(1)
        else:
            predicted_win_loss.append(0)

rugby['predicted_points_diff'] = predicted_points_diff
rugby['predicted_win_loss'] = predicted_win_loss

win_accuracy = []

for x in x_list:
    if rugby.game[x] < 67:
        if rugby.predicted_win_loss[x] == 1:
            if rugby.predicted_win_loss[x] == rugby.win[x]:
                win_accuracy.append(1)

sum(win_accuracy)

##########################################################################
### function for testing different home field advantages ###
##########################################################################

def homeAdvantage(number):
    
    x_list = list(range(0,240))
    
    prediction_homeadjusted = []
    
    for x in x_list:
        if rugby.home[x] == 1:
            prediction_homeadjusted.append(rugby.predicted_points_diff[x] + number)
        else:
            prediction_homeadjusted.append(rugby.predicted_points_diff[x])
    
    rugby['prediction_homeadjusted'] = prediction_homeadjusted
    
    win_loss_homeadjusted = []
    points_diff_homeadjusted =[]
    
    for x in x_list:
        if x == 0:
            expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x+1])
            points_diff_homeadjusted.append(expected)
            if expected > 0:
                win_loss_homeadjusted.append(1)
            else:
                win_loss_homeadjusted.append(0)
        elif 0 < x < 239:
            if rugby.game[x] == rugby.game[x+1]:
                expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x+1])
                points_diff_homeadjusted.append(expected)
                if expected > 0:
                    win_loss_homeadjusted.append(1)
                else:
                    win_loss_homeadjusted.append(0)
            if rugby.game[x] == rugby.game[x-1]:
                expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x-1])
                points_diff_homeadjusted.append(expected)
                if expected > 0:
                    win_loss_homeadjusted.append(1)
                else:
                    win_loss_homeadjusted.append(0)
        elif x == 239:
            expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x-1])
            points_diff_homeadjusted.append(expected)
            if expected > 0:
                win_loss_homeadjusted.append(1)
            else:
                win_loss_homeadjusted.append(0)
    
    rugby['win_loss_homeadjusted'] = win_loss_homeadjusted
    
    win_accuracy_home = []
    
    for x in x_list:
        if rugby.game[x] < 67:
            if rugby.win_loss_homeadjusted[x] == 1:
                if rugby.win_loss_homeadjusted[x] == rugby.win[x]:
                    win_accuracy_home.append(1)
    
    print(float(sum(win_accuracy_home)))
    
    error_home = []
    
    for x in x_list:
        if x == 0:
            expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x+1])
            error = expected - rugby.points_diff[x]
            error_sq = error ** 2            
            error_home.append(error_sq)            
        elif 0 < x < 239:
            if rugby.game[x] == rugby.game[x+1]:
                expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x+1])
                error = expected - rugby.points_diff[x]
                error_sq = error ** 2            
                error_home.append(error_sq)
            if rugby.game[x] == rugby.game[x-1]:
                expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x-1])
                error = expected - rugby.points_diff[x]
                error_sq = error ** 2            
                error_home.append(error_sq)
        elif x == 239:
            expected = (rugby.prediction_homeadjusted[x] - rugby.prediction_homeadjusted[x-1])
            error = expected - rugby.points_diff[x]
            error_sq = error ** 2            
            error_home.append(error_sq)
            
    print(float(sum(error_home)))
    

##### a really bad way to do it, but figuring out the best value for home field advantage

deci = [5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1]

x_deci = list(range(0,14))

for x in x_deci:
    print deci[x]
    print homeAdvantage(deci[x])

centi = [5.80,5.81,5.82,5.83,5.84,5.85,5.86,5.87,5.88,5.89,
         5.90,5.91,5.92,5.93,5.94,5.95,5.96,5.97,5.98,5.99]

x_centi = list(range(len(centi)))        

for x in x_centi:
    print centi[x]
    print homeAdvantage(centi[x])

### when using point differential as "strength," the model is improved by
### incorporating a home-field advantage of about 5.95 points

##########################################################################
###  ###
##########################################################################


feature_cols = ['home','avgPD','r11_opp_pd']

X = rugby[rugby.round < 12][feature_cols]
y = rugby[rugby.round < 12].predicted_win_loss

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)


from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
metrics.confusion_matrix(y_test,y_pred)


y_prob = logreg.predict_proba(x_test)[:,1]

metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)