"""
Created on Thu Apr 23 21:38:27 2015

@author: Trevor1
"""

#Imports#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import seaborn as sns

rugby = pd.read_table('aviva_fixed.csv', sep=',') # import csv sheet

'''I need to convert a column (tackles)
because it has two data points in it: tackles made and those missed
I'll do this by splicing everything before the '/' into a tackles made column
and everything after it into a tackles missed column
then I'll simply get rid of that column'''

## first create two empty lists for the new columns

tackles_made = []
tackles_missed = []

## define what those lists will include by splicing the old column

for row in rugby.tackles:
    spot1 = row.find('/')
    spot2 = (row.find('/')) + 1
    tackles_made.append(int(row[0:spot1]))
    tackles_missed.append(int(row[spot2:]))

## assign two new columns and tell python what to populate them with

rugby['tackles_made'] = tackles_made
rugby['tackles_missed'] = tackles_missed

del rugby['tackles']

''' Now I'll do the same process as above, except for the conversions column'''

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

'''and again for the mauls column'''

mauls_won = []
mauls_started = []
maul_success = []

for row in rugby.mauls:
    spot1 = row.find(' from ')
    spot2 = spot1 + 6
    spot3 = (row.find('(')) + 1
    spot4 = spot3 - 2
    if '(' in row:
        mauls_won.append(float(row[0:spot1]))
        mauls_started.append(float(row[spot2:spot4]))
        maul_success.append(float(row[spot3:-2]))
    else:
        maul_success.append(100.0)         ##if no mauls were started, teams are given 100% success
        mauls_won.append(float(row[0:spot1]))
        mauls_started.append(float(row[spot2:]))
 
rugby['mauls_won'] = mauls_won
rugby['mauls_started'] = mauls_started
rugby['maul_success'] = maul_success

del rugby['mauls']

'''and again for the rucks'''

rucks_won = []
rucks_started = []
ruck_success = []

for row in rugby.rucks:
    spot1 = row.find(' from ')
    spot2 = spot1 + 6
    spot3 = (row.find('(')) + 1
    spot4 = spot3 - 2
    if '(' in row:
        rucks_won.append(float(row[0:spot1]))
        rucks_started.append(float(row[spot2:spot4]))
        ruck_success.append(float(row[spot3:-2]))
    else:
        ruck_success.append(100.0)         ##if no rucks were started, teams are given 100% success
        rucks_won.append(float(row[0:spot1]))
        rucks_started.append(float(row[spot2:]))

rugby['rucks_won'] = rucks_won
rugby['rucks_started'] = rucks_started
rugby['ruck_success'] = ruck_success

del rugby['rucks']

'''and again for scrums'''

scrums_lost = []
scrums_won = []
scrum_success = []

for row in rugby.scrums:
    spot1 = row.find(' won, ')
    spot2 = spot1 + 6
    spot3 = (row.find('(')) + 1
    spot4 = spot3 - 7
    scrums_won.append(float(row[0:spot1]))
    scrums_lost.append(float(row[spot2:spot4]))
    scrum_success.append(float(row[spot3:-2]))

rugby['scrums_won'] = mauls_won
rugby['scrums_lost'] = mauls_started
rugby['scrum_success'] = maul_success

del rugby['scrums']

rugby.head()

'''and again for lineouts'''

lo_lost = []
lo_won = []
lo_success = []

for row in rugby.lineouts:
    spot1 = row.find(' won, ')
    spot2 = spot1 + 6
    spot3 = (row.find('(')) + 1
    spot4 = spot3 - 7
    lo_won.append(float(row[0:spot1]))
    lo_lost.append(float(row[spot2:spot4]))
    lo_success.append(float(row[spot3:-2]))

rugby['lo_won'] = lo_won
rugby['lo_lost'] = lo_lost
rugby['lo_success'] = lo_success

del rugby['lineouts']

'''and again to split penalties and free kicks'''

penkicks = []
freekicks = []

for row in rugby.penalties:
    spot1 = row.find(' (')
    if '(' in row:
        penkicks.append(float(row[0:spot1]))
        freekicks.append(float(row[-2]))
    else:
        penkicks.append(float(row))
        freekicks.append(0)
    
rugby['penkicks'] = penkicks
rugby['freekicks'] = freekicks

del rugby['penalties']

''' and again to determine penalty goes made/missed '''

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

''' now I will make a points column '''

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

'''meters per run column'''

rugby['meters_per_run'] = rugby.meters_run / rugby.runs

''' points_against column '''

for row in rugby:
    rugby[rugby.game == ]

### Visualizations ###

pd.scatter_matrix(rugby)
plt.show()

rugby.groupby('team').points.mean().plot(kind='bar', title='Average Number of Points per Game')
plt.show()

rugby.boxplot(column='pen_goals',by='win')
plt.xlabel('Win/Loss')
plt.ylabel('Penalty Goals')
plt.show()

rugby.boxplot(column='tries',by='win')
plt.xlabel('Win/Loss')
plt.ylabel('Tries')
plt.show()

rugby.boxplot(column='conversions_made',by='win')
plt.xlabel('Win/Loss')
plt.ylabel('Conversions Made')
plt.show()

sns.lmplot(x='passes', y='meters_run', data=rugby)
sns.lmplot(x='clean_breaks', y='defenders_beat', data=rugby)

sns.lmplot(x='passes', y='points', data=rugby, order=2) #positive
sns.lmplot(x='meters_run', y='points', data=rugby, order=2) #positive
sns.lmplot(x='clean_breaks', y='points', data=rugby, order=2) #positive
sns.lmplot(x='defenders_beat', y='points', data=rugby, order=2) #positive
sns.lmplot(x='rucks_won', y='points', data=rugby, order=2) #slightly negative
sns.lmplot(x='rucks_started', y='points', data=rugby, order=2) #slightly negative
sns.lmplot(x='tackles_missed', y='points', data=rugby, order=2) #useless
sns.lmplot(x='tackles_made', y='points', data=rugby, order=2) #useless
sns.lmplot(x='scrums_won', y='points', data=rugby, order=2) #useless
sns.lmplot(x='turns_conceded', y='points', data=rugby, order=2) #negative
sns.lmplot(x='offloads', y='points', data=rugby, order=2) #positive
sns.lmplot(x='kicks_from_hand', y='points', data=rugby, order=2) #pretty useless
sns.lmplot(x='runs', y='points', data=rugby, order=2) #positive
sns.lmplot(x='meters_per_run', y='points', data=rugby, order=2) #positive
sns.lmplot(x='tackle_success', y='points', data=rugby, order=2) #positive

### 






## Team DataFrames ##

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

