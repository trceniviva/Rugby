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

## naming feature columns

feature_cols = ['sale', 'irish', 'exeter', 'wasps', 'northampton', 'yorkshire',
                'newcastle', 'harlequins', 'bath', 'gloucester', 'leicester',
                'saracens', 'worcester', 'welsh','sale_home', 'irish_home', 'exeter_home', 'wasps_home',
                'northampton_home', 'yorkshire_home', 'newcastle_home',
                'harlequins_home', 'bath_home', 'gloucester_home', 'leicester_home',
                'saracens_home', 'worcester_home', 'welsh_home','sale_opp', 'irish_opp', 'exeter_opp', 'wasps_opp',
                'northampton_opp', 'yorkshire_opp', 'newcastle_opp',
                'harlequins_opp', 'bath_opp', 'gloucester_opp', 'leicester_opp',
                'saracens_opp', 'worcester_opp', 'welsh_opp']
                
###########################################################################
########################## LOGISTIC #######################################
######################### REGRESSION ######################################
###########################################################################

## training on the first 15 rounds of each season

x_train = rugby[rugby['1011'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1011'] == 1][rugby.round < 16].win

## testing on the 16th through 22nd rounds

x_test = rugby[rugby['1011'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1011'] == 1][rugby.round > 15].win

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

## repeat for the next season in my dataset

x_train = rugby[rugby['1112'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1112'] == 1][rugby.round < 16].win

x_test = rugby[rugby['1112'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1112'] == 1][rugby.round > 15].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

######################### 

x_train = rugby[rugby['1213'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1213'] == 1][rugby.round < 16].win

x_test = rugby[rugby['1213'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1213'] == 1][rugby.round > 15].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

######################### 

x_train = rugby[rugby['1314'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1314'] == 1][rugby.round < 16].win

x_test = rugby[rugby['1314'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1314'] == 1][rugby.round > 15].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

######################### 

x_train = rugby[rugby['1415'] == 1][rugby.round < 16][feature_cols]
y_train = rugby[rugby['1415'] == 1][rugby.round < 16].win

x_test = rugby[rugby['1415'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1415'] == 1][rugby.round > 15].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

#########################

x_train = rugby[rugby.round < 16][feature_cols]
y_train = rugby[rugby.round < 16].win

x_test = rugby[rugby.round > 15][feature_cols]
y_test = rugby[rugby.round > 15].win

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test,y_pred)
logreg.coef_

y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)
metrics.log_loss(y_test,y_prob)

x_test = rugby[rugby['1011'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1011'] == 1][rugby.round > 15].win
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)  
y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)

x_test = rugby[rugby['1112'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1112'] == 1][rugby.round > 15].win
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred) 
y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob) 

x_test = rugby[rugby['1213'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1213'] == 1][rugby.round > 15].win
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)
y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)  

x_test = rugby[rugby['1314'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1314'] == 1][rugby.round > 15].win
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)
y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)  

x_test = rugby[rugby['1415'] == 1][rugby.round > 15][feature_cols]
y_test = rugby[rugby['1415'] == 1][rugby.round > 15].win
y_pred = logreg.predict(x_test)
metrics.accuracy_score(y_test, y_pred)
y_prob = logreg.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test,y_prob)  


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

knn = KNeighborsClassifier(n_neighbors=21)
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
