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
        del rugby[elem + '_opp']
    rugby['game'] = rugby.index + 1
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_points1'] = rugby[str(elem) + '_home'] * rugby['points']
        rugby[str(elem) + '_points2'] = (-1)(rugby[str(elem) + '_home'] - 1) * rugby['points_against']
    rugby.to_csv('clean' + str(df))


rugbyDFs = ['rugby1011.csv','rugby1112.csv','rugby1213.csv','rugby1314.csv','rugby1415.csv']

for df in rugbyDFs:
    feature_maker(df)




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
    if 'worcester' in rugby:
        rugby['worcester'] = rugby['worcester'] + rugby['worcester_opp']
        del rugby['worcester_opp']
    rugby['saracens'] = rugby['saracens'] + rugby['saracens_opp']
    del rugby['saracens_opp']
    if 'newcastle' in rugby:    
        rugby['newcastle'] = rugby['newcastle'] + rugby['newcastle_opp']
        del rugby['newcastle_opp']
    rugby['northampton'] = rugby['northampton'] + rugby['northampton_opp']
    del rugby['northampton_opp']
    if 'welsh' in rugby:
        rugby['welsh'] = rugby['welsh'] + rugby['welsh_opp']
        del rugby['welsh_opp']
    if 'yorkshire' in rugby:
        rugby['yorkshire'] = rugby['yorkshire'] + rugby['yorkshire_opp']
        del rugby['yorkshire_opp']


    ## creating win percent features
    rugby['win_percent'] = (rugby['wins'] / (rugby['round'] - 1))
    rugby['opp_win_percent'] = (rugby['wins_opp'] / (rugby['round'] - 1))
    rugby['win_percent'] = rugby['win_percent'].fillna(0)
    rugby['opp_win_percent'] = rugby['opp_win_percent'].fillna(0)
    
    ## creating dummy variables for seasons
    for elem in rugby['season'].unique():
        rugby['season_' + str(elem)] = rugby['season'] == elem
    seasons = ('season_' + rugby['season'].unique())
    rugby[seasons] = rugby[seasons].astype(int)
    
    ## creating wins columns
    for elem in rugby['team'].unique():
        rugby[str(elem) + '_wins'] = rugby[str(elem)] * rugby[str(elem) + '_home'] * rugby['wins']
        rugby[str(elem) + '_Oppwins'] = rugby[str(elem)] * rugby[str(elem) + '_home'] * rugby['wins']
