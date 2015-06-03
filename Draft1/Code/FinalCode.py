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

rugby = pd.read_table('dataFINAL.csv', sep=',')

###########################################################################
########################## VISUALIZATIONS #################################
###########################################################################

teams = ['sale', 'irish', 'exeter', 'wasps', 'northampton', 'yorkshire',
                'newcastle', 'harlequins', 'bath', 'gloucester', 'leicester',
                'saracens', 'worcester', 'welsh']

for elem in teams:
    rugby[elem] = rugby[elem] - rugby[str(elem) + '_home']

## naming feature columns

feature_cols = ['sale_home', 'irish_home', 'exeter_home',
                'wasps_home', 'northampton_home', 'yorkshire_home',
                'newcastle_home', 'harlequins_home', 'bath_home', 'gloucester_home', 'leicester_home',
                'saracens_home', 'worcester_home', 'welsh_home','sale_opp', 'irish_opp', 'exeter_opp', 'wasps_opp', 'northampton_opp', 'yorkshire_opp',
                'newcastle_opp', 'harlequins_opp', 'bath_opp', 'gloucester_opp', 'leicester_opp',
                'saracens_opp', 'worcester_opp', 'welsh_opp']
                

                
###########################################################################
########################## LOGISTIC #######################################
######################### REGRESSION ######################################
###########################################################################

## training on the first 15 rounds of each season

x_train = rugby[rugby['1011'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1011'] == 1][rugby.round < 16].points_diff

## testing on the 16th through 22nd rounds

x_test = rugby[rugby['1011'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1011'] == 1][rugby.round > 15].points_diff

## fit the model

logreg.fit(x_train, y_train)

## create predictions

y_pred = logreg.predict(x_test)

## test accuracy

metrics.accuracy_score(y_test,y_pred)
logreg.coef_

# determine probability

y_prob = logreg.predict_proba(x_test)[:,1]

# test accuracy 

metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

######################### 



def team_strengths(season_str,round_int):  
    
    x_train = rugby[rugby[str(season_str)] == 1][rugby.round < round_int][feature_cols]
    y_train = rugby[rugby[str(season_str)] == 1][rugby.round < round_int].win
    
    x_test = rugby[rugby[str(season_str)] == 1][rugby.round == round_int][feature_cols]
    y_test = rugby[rugby[str(season_str)] == 1][rugby.round == round_int].win
    
    logreg.fit(x_train, y_train)
    
    y_pred = logreg.predict(x_test)
    metrics.accuracy_score(y_test,y_pred) 
    
    pred_list = y_pred
    test_list = list(y_test)
    errors = pred_list - test_list
    sqerrors = []
    for x in errors:
        sqerrors.append(x ** 2)
    error = sum(sqerrors) /  len(sqerrors)
    error = error ** .5
    print metrics.accuracy_score(y_test,y_pred)
    accuracies.append(metrics.accuracy_score(y_test,y_pred))    
    print 'For season ' + str(season_str) + ' round ' + str(round_int) + ','
    print 'the expected strengths of each team are:'
    print 'Sale:'
    print logreg.coef_[round_int][0] * (-1)
    print 'Irish:'
    print logreg.coef_[round_int][1] * (-1)
    print 'Exeter:'
    print logreg.coef_[round_int][2] * (-1)
    print 'Wasps:'
    print logreg.coef_[round_int][3] * (-1)
    print 'Northampton:'
    print logreg.coef_[round_int][4] * (-1)
    print 'Yorkshire:'
    print logreg.coef_[round_int][5] * (-1) 
    print 'Newcastle:'
    print logreg.coef_[round_int][6] * (-1)
    print 'Harlequins:'
    print logreg.coef_[round_int][7] * (-1)
    print 'Bath:'
    print logreg.coef_[round_int][8] * (-1)
    print 'Gloucester:'
    print logreg.coef_[round_int][9] * (-1)
    print 'Leicester:'
    print logreg.coef_[round_int][10] * (-1)
    print 'Saracens:'
    print logreg.coef_[round_int][11] * (-1)
    print 'Worcester:'
    print logreg.coef_[round_int][12] * (-1)
    print 'Welsh:'
    print logreg.coef_[round_int][13] * (-1)
      
    
seasons = ['1011','1112','1213','1314','1415']
rounds = [15, 16, 17, 18, 19, 20, 21, 22]
accuracies = []  

for x in seasons:    
    for y in rounds:
        team_strengths(x,y)

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

######################### 

x_train = rugby[rugby['1213'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1213'] == 1][rugby.round < 16].points_diff

x_test = rugby[rugby['1213'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1213'] == 1][rugby.round > 15].points_diff

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

######################### 

x_train = rugby[rugby['1314'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1314'] == 1][rugby.round < 16].points_diff

x_test = rugby[rugby['1314'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1314'] == 1][rugby.round > 15].points_diff

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

######################### 

x_train = rugby[rugby['1415'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1415'] == 1][rugby.round < 16].points_diff

x_test = rugby[rugby['1415'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1415'] == 1][rugby.round > 15].points_diff

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

#########################

'''
Now I'm looking at using the first 15 rounds from all the seasons
in order to predict the outcome of the rest of those seasons.
Additionally, once I've trained on the first 15 rounds of all seasons,
I'll use that model to predict the second half of each season individually.
'''

x_train = rugby[rugby.round < 16][feature_cols]
y_train = rugby[rugby.round < 16].points_diff

x_test = rugby[rugby.round > 15][feature_cols]
y_test = rugby[rugby.round > 15].points_diff

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

x_test = rugby[rugby['1011'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1011'] == 1][rugby.round > 15].points_diff
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)  

x_test = rugby[rugby['1112'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1112'] == 1][rugby.round > 15].points_diff
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred) 

x_test = rugby[rugby['1213'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1213'] == 1][rugby.round > 15].points_diff
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)

x_test = rugby[rugby['1314'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1314'] == 1][rugby.round > 15].points_diff
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)

x_test = rugby[rugby['1415'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1415'] == 1][rugby.round > 15].points_diff
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)

###########################################################################
########################## K-NEAREST ######################################
########################### NEIGHBOR ######################################
###########################################################################

x_train = rugby[rugby.round < 16][feature_cols]
y_train = rugby[rugby.round < 16].win

x_test = rugby[rugby.round > 15][feature_cols]
y_test = rugby[rugby.round > 15].win

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
metrics.accuracy_score(y_test, y_pred)  

k_list = []

knn = KNeighborsClassifier(n_neighbors= 19)
knn.fit(x_train, y_train)

x_test = rugby[rugby['1011'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1011'] == 1][rugby.round > 15].win
y_pred = knn.predict(x_test)
k_list.append(metrics.accuracy_score(y_test, y_pred))  

x_test = rugby[rugby['1112'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1112'] == 1][rugby.round > 15].win
y_pred = knn.predict(x_test)
k_list.append(metrics.accuracy_score(y_test, y_pred))  

x_test = rugby[rugby['1213'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1213'] == 1][rugby.round > 15].win
y_pred = knn.predict(x_test)
k_list.append(metrics.accuracy_score(y_test, y_pred))  

x_test = rugby[rugby['1314'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1314'] == 1][rugby.round > 15].win
y_pred = knn.predict(x_test)
k_list.append(metrics.accuracy_score(y_test, y_pred))  

x_test = rugby[rugby['1415'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1415'] == 1][rugby.round > 15].win
y_pred = knn.predict(x_test)
k_list.append(metrics.accuracy_score(y_test, y_pred))  

ksum = sum(k_list)
klen = len(k_list)
ksum / klen

###########################################################################
########################## NEXT ######################################
########################### MODELS ######################################
###########################################################################

from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor()
from sklearn.cross_validation import cross_val_score

x_train = rugby[rugby.round < 16][feature_cols]
y_train = rugby[rugby.round < 16].points_diff

x_test = rugby[rugby.round > 15][feature_cols]
y_test = rugby[rugby.round > 15].points_diff

treereg.fit(x_train,y_train)

scores = cross_val_score(treereg, x_train, y_train, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

# define a range of values
max_depth_range = range(1, 11)

# create an empty list to store the average RMSE for each value of max_depth
RMSE_scores = []

for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth)
    MSE_scores = cross_val_score(treereg, x_train, y_train, cv=10, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
RMSE_scores

treereg = DecisionTreeRegressor()
treereg.fit(x_train,y_train)
y_pred = treereg.predict(x_test)
scores = cross_val_score(treereg, x_train, y_train, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

######################################################################
########################## NEXT ######################################
########################### MODELS ###################################
######################################################################

feature_cols = ['sale', 'irish', 'exeter', 'wasps', 'northampton', 'yorkshire',
                'newcastle', 'harlequins', 'bath', 'gloucester', 'leicester',
                'saracens', 'worcester', 'welsh']

x_train = rugby[rugby['1011'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1011'] == 1][rugby.round < 16].points_diff

logreg.fit(x_train, y_train)
logreg.coef_

x_test = rugby[rugby['1011'] == 1][rugby.round == 16][feature_cols]
y_test = rugby[rugby['1011'] == 1][rugby.round == 16].points_diff

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
