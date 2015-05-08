# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:48:36 2015

@author: Trevor1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import seaborn as sns

########################################################################

rugby = pd.read_table('aviva_fixed.csv', sep=',') # import csv sheet

########################################################################

conversions_made = []
conversions_missed = []

for row in rugby.conversions:
    spot1 = row.find(' from ')
    spot2 = spot1 + 6
    conversions_made.append(int(row[0:spot1]))
    conversions_missed.append(int(row[spot2:]))

rugby['conversions_made'] = conversions_made
rugby['conversions_missed'] = conversions_missed

del rugby['conversions']

pen_goals = []
pen_attempt = []

for row in rugby.pens:
    pen_goals.append(float(row[0]))
    pen_attempt.append(float(row[-1]))

rugby['pen_goals'] = pen_goals
rugby['pen_attempt'] = pen_attempt

del rugby['pens']

''' and a similar process to clean up the drop_goals category '''

drop_goals = []
drop_attempt = []

for row in rugby.drop_goals:
    if '(' in row:
        drop_goals.append(float(row[0]))
        drop_attempt.append(float(row[3]))
    else:
        drop_goals.append(0.0)
        drop_attempt.append(0.0)

rugby['drop_goals'] = drop_goals
rugby['drop_attempt'] = drop_attempt

''' and lastly for the tries column '''

tries = []

for row in rugby.tries:
    if '(' in row:
        tries.append(float(row[0]))
    else:
        tries.append(float(row))

rugby['tries'] = tries

########################################################################

try_points = []

for game in tries:
    try_points.append(game*5)

rugby['try_points'] = try_points

con_points = []

for game in conversions_made:
    con_points.append(game*2)

rugby['con_points'] = con_points

pen_points = []

for game in pen_goals:
    pen_points.append(game*3)

rugby['pen_points'] = pen_points

rugby['points'] = rugby.try_points + rugby.con_points + rugby.pen_points

########################################################################

points = rugby.points
points = list(points)

game = rugby.game
game = list(game)

win = rugby.win
win = list(win)

points_diff = []

x_list = list(range(0,240))

for x in x_list:
    if x < 239:        
        if game[x] == game[x+1]:
            points_diff.append((points[x]) - (points[x+1]))
            points_diff.append((-1)*((points[x]) - (points[x+1])))

rugby['points_diff'] = points_diff

rugby['points_against'] = rugby.points - rugby.points_diff

########################################################################

teams = rugby.team
teams = list(teams)

opponent = []

for x in x_list:
    if x < 239:        
        if game[x] == game[x+1]:
            opponent.append(teams[x+1])
            opponent.append(teams[x])

rugby['opponent'] = opponent

########################################################################

rugby = rugby[['season', 'game', 'round', 'team','home',
       'points','try_points','con_points','pen_points',
       'points_diff','points_against','opponent','win','draw']]

########################################################################

bath_df = rugby[rugby.team == 'bath']
northampton_df = rugby[rugby.team == 'northampton']
gloucester_df = rugby[rugby.team == 'gloucester']
irish_df = rugby[rugby.team == 'irish']
harlequins_df = rugby[rugby.team == 'harlequins']
leicester_df = rugby[rugby.team == 'leicester']
newcastle_df = rugby[rugby.team == 'newcastle']
saracens_df = rugby[rugby.team == 'saracens']
wasps_df = rugby[rugby.team == 'wasps']
sale_df = rugby[rugby.team == 'sale']
welsh_df = rugby[rugby.team == 'welsh']
exeter_df = rugby[rugby.team == 'exeter']

########################################################################

### resetting the index for all of these data frames
### in order to have a consecutive index corresponding with 'round'

bath = bath_df.set_index(['round'], drop=False)
northampton = northampton_df.set_index(['round'], drop=False)
gloucester = gloucester_df.set_index(['round'], drop=False)
irish = irish_df.set_index(['round'],drop=False)
harlequins = harlequins_df.set_index(['round'], drop=False)
leicester = leicester_df.set_index(['round'], drop=False)
newcastle = newcastle_df.set_index(['round'], drop=False)
saracens = saracens_df.set_index(['round'], drop=False)
wasps = wasps_df.set_index(['round'],drop=False)
sale = sale_df.set_index(['round'],drop=False)
welsh = welsh_df.set_index(['round'],drop=False)
exeter = exeter_df.set_index(['round'],drop=False)

########################################################################

def avgPD(team):
    x_rounds = list(range(1,21))
    avgPD = []
    for x in x_rounds:
        avgPD.append((sum(team.points_diff[0:x]))/x) 
    team['avgPD'] = avgPD
    team.avgPD

avgPD(bath)
avgPD(northampton)
avgPD(gloucester)
avgPD(irish)
avgPD(harlequins)
avgPD(leicester)
avgPD(newcastle)
avgPD(saracens)
avgPD(wasps)
avgPD(sale)
avgPD(welsh)
avgPD(exeter)

########################################################################

def avgPD_opp(team):
    x_rounds = list(range(1,21))
    avgPD_opp = []
    if team.opponent[x] == 'sale':
        avgPD_opp.append(sale.avgPD[x])
    elif team.opponent[x] == 'welsh':
        avgPD_opp.append(welsh.avgPD[x])
    elif team.opponent[x] == 'leicester':
        avgPD_opp.append(leicester.avgPD[x])
    elif team.opponen[x] == 'northampton':
        avgPD_opp.append(northampton.avgPD[x])
    elif team.opponent[x] == 'saracens':
        avgPD_opp.append(saracens.avgPD[x])
    elif team.opponent[x] == 'wasps':
        avgPD_opp.append(wasps.avgPD[x])
    elif team.opponent[x] == 'newcastle':
        avgPD_opp.append(newcastle.avgPD[x])
    elif team.opponent[x] == 'irish':
        avgPD_opp.append(irish.avgPD[x])
    elif team.opponent[x] == 'harlequins':
        avgPD_opp.append(harlequins.avgPD[x])
    elif team.opponent[x] == 'gloucester':
        avgPD_opp.append(gloucester.avgPD[x])
    elif team.opponent[x] == 'exeter':
        avgPD_opp.append(exeter.avgPD[x])
    avgPD_opp
    
    team['avgPD_opp'] = avgPD_opp