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

match = ['match']

rugby = pd.read_table('rugby1112.csv', names=match, sep=',')

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
    points.append(point)
    stop3 = stop2 + 3
    backends.append(x[stop3:])

for x in backends:
    stop = x.find('\xca')
    points_ag = x[:stop]
    points_against.append(points_ag)
    stop2 = stop + 1
    opp = x[stop2:]
    opponent.append(opp)    

points_against    
opponent

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

rugby['team']= teams
rugby['points'] = points
rugby['points_against'] = points_against
rugby['opponent'] = opps
del rugby['match']

rugby.to_csv('1112_clean.csv')