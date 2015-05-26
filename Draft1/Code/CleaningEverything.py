# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:28:22 2015

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

def feature_maker(df):
    match = ['match']
    rugby = pd.read_table(str(df), names=match, sep=',')
    rugby
    matches = list(rugby.match)
    teamlist = []
    points = []
    points_against = []
    opponent = []
    backends = []
    for x in matches:
        stop = x.find('\xca')
        home = x[:stop]
        teamlist.append(home)
        stop2 = x.find(' - ')
        stop = stop + 1
        point = x[stop:stop2]
        points.append(float(point))
        stop3 = stop2 + 3
        backends.append(x[stop3:])
    for x in backends:
        stop = x.find('\xca')
        points_ag = x[:stop]
        points_against.append(float(points_ag))
        stop2 = stop + 1
        opp = x[stop2:]
        opponent.append(opp)    
    teams = []
    for team in teamlist:
        if team == 'London Irish':
            teams.append('irish')
        elif team == 'Northampton Saints':
            teams.append('northampton')
        elif team == 'Exeter Chiefs':
            teams.append('exeter')
        elif team == 'Sale Sharks':
            teams.append('sale')
        elif team == 'Saracens':
            teams.append('saracens')
        elif team == 'Wasps':
            teams.append('wasps')
        elif team == 'Worcester Warriors':
            teams.append('worcester')
        elif team == 'Newcastle Falcons':
            teams.append('newcastle')
        elif team == 'Gloucester Rugby':
            teams.append('gloucester')
        elif team == 'Leicester Tigers':
            teams.append('leicester')
        elif team == 'Harlequins':
            teams.append('harlequins')
        elif team == 'Bath Rugby':
            teams.append('bath')
        elif team == 'London Welsh':
            teams.append('welsh')
        elif team == 'Yorkshire Carnegie':
            teams.append('yorkshire')
    opps = []
    for team in opponent:
        if team == 'London Irish':
            opps.append('irish')
        elif team == 'Northampton Saints':
            opps.append('northampton')
        elif team == 'Exeter Chiefs':
            opps.append('exeter')
        elif team == 'Sale Sharks':
            opps.append('sale')
        elif team == 'Saracens':
            opps.append('saracens')
        elif team == 'Wasps':
            opps.append('wasps')
        elif team == 'Worcester Warriors':
            opps.append('worcester')
        elif team == 'Newcastle Falcons':
            opps.append('newcastle')
        elif team == 'Gloucester Rugby':
            opps.append('gloucester')
        elif team == 'Leicester Tigers':
            opps.append('leicester')
        elif team == 'Harlequins':
            opps.append('harlequins')
        elif team == 'Bath Rugby':
            opps.append('bath')
        elif team == 'London Welsh':
            opps.append('welsh')
        elif team == 'Yorkshire Carnegie':
            opps.append('yorkshire')
    rugby['team']= teams
    rugby['points'] = points
    rugby['points_against'] = points_against
    rugby['opponent'] = opps
    del rugby['match']
    rugby['points_diff'] = rugby['points'] - rugby['points_against']
    rugby['win'] = (rugby['points_diff'] > 0).astype(int)
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
    for elem in rugby['team'].unique():
        rugby[elem] = rugby[elem] + rugby[elem + '_opp']
    ## setting the 'game'
    rugby['game'] = rugby.index + 1
    negatives = [-1] * len(rugby)
    rugby['negatives'] = negatives
    ## filling in columns for points scored
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_points1'] = rugby[str(elem) + '_home'] * rugby['points']
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_points2'] = ((rugby[str(elem) + '_home'] - rugby[elem]) * rugby['points_against']) * rugby['negatives']
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_points'] = rugby[str(elem) + '_points1'] + rugby[str(elem) + '_points2']
        del rugby[elem + '_points1']
        del rugby[elem + '_points2']
    # CREATING A WIN COLUMN FOR EACH TEAM
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_win1'] = rugby[str(elem) + '_home'] * rugby['win']
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_win2'] = ((rugby[str(elem) + '_home'] - rugby[elem]) * rugby['win']) * rugby['negatives']
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_win'] = rugby[str(elem) + '_win1'] + rugby[str(elem) + '_win2']
        del rugby[elem + '_win1']
        del rugby[elem + '_win2']
    rugby['round'] = (((rugby['game'] - 1) / 6) + 1)
    rugby['round'] = rugby['round'].astype(int)
    df = df[-13:]
    rugby['season'] = df[5:9]
    for elem in rugby['season'].unique():
        rugby[str(elem)] = rugby['season'] == elem
        rugby[str(elem)] = rugby[str(elem)].astype(int)
    del rugby['season']
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_win'] = rugby[str(elem)] * rugby['win']
    rugby.to_csv('clean' + str(df))


rugbyDFs = ['rugby1011.csv',
            'rugby1112.csv',
            'rugby1213.csv',
            'rugby1314.csv',
            'rugby1415.csv']

for df in rugbyDFs:
    feature_maker(df)

cleanDFs = ['cleanrugby1011.csv',
            'cleanrugby1112.csv',
            'cleanrugby1213.csv',
            'cleanrugby1314.csv',
            'cleanrugby1415.csv']

list_ = []

for item in cleanDFs:
    df = pd.read_csv(str(item))
    list_.append(df)
frame = pd.concat(list_)
frame = frame.fillna(0)
del frame['Unnamed: 0']

frame.to_csv('dataALMOST.csv')

rugby = pd.read_table('dataALMOST.csv', sep=',')

for elem in rugby['team'].unique():
    rugby[str(elem) + '_new_opp'] = rugby[str(elem)]
    rugby[str(elem) + '_new_team'] = rugby[str(elem) + '_opp']
    rugby[str(elem) + '_opp'] = rugby[str(elem) + '_new_opp']
    rugby[str(elem)] = rugby[str(elem) + '_new_team']
    del rugby[str(elem) + '_new_team']
    del rugby[str(elem) + '_new_opp']

rugby['win_fix'] = (rugby['win'] * rugby['negatives']) + 1
rugby['win'] = rugby['win_fix']
del rugby['win_fix']

rugby['points_diff'] = rugby['points_diff'] * rugby['negatives']

rugby.to_csv('data2.csv')

fullDFs = ['dataALMOST.csv','data2.csv']

list_ = []

for item in fullDFs:
    df = pd.read_csv(str(item))
    list_.append(df)
frame = pd.concat(list_)
frame = frame.fillna(0)
del frame['Unnamed: 0']

frame.to_csv('dataFINAL.csv')

rugby = pd.read_table('dataFINAL.csv',sep=',')