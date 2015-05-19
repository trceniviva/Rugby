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

''' points_against and points_diff column '''

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

'''opponent column'''

teams = rugby.team
teams = list(teams)

opponent = []

for x in x_list:
    if x < 239:        
        if game[x] == game[x+1]:
            opponent.append(teams[x+1])
            opponent.append(teams[x])

rugby['opponent'] = opponent

### adding a column that indicates the teams average point differential throughout the season

bath_pd = 9.20
north_pd = 9.15
glou_pd = -1.35
irish_pd = -5.70
quins_pd = -2.80
leic_pd = .95
new_pd = -3.85
sar_pd = 9.95
wasp_pd = 7.80
sale_pd = 1.85
welsh_pd = -32.80
exeter_pd = 7.60

error_avg_pd = [bath_pd,north_pd,glou_pd,
                irish_pd,quins_pd,leic_pd,
                new_pd,sar_pd,wasp_pd,
                sale_pd,welsh_pd,exeter_pd]

x_list = list(range(0,240))

r11_avg_pd = []
r11_opp_pd = []

for x in x_list:
    if rugby.round[x] < 12:
        if rugby.team[x] == 'bath':
            r11_avg_pd.append(bath_pd)
        elif rugby.team[x] == 'northampton':
            r11_avg_pd.append(north_pd)
        elif rugby.team[x] == 'gloucester':
            r11_avg_pd.append(glou_pd)
        elif rugby.team[x] == 'irish':
            r11_avg_pd.append(irish_pd)
        elif rugby.team[x] == 'harlequins':
            r11_avg_pd.append(quins_pd)
        elif rugby.team[x] == 'leicester':
            r11_avg_pd.append(leic_pd)
        elif rugby.team[x] == 'newcastle':
            r11_avg_pd.append(new_pd)
        elif rugby.team[x] == 'saracens':
            r11_avg_pd.append(sar_pd)
        elif rugby.team[x] == 'wasps':
            r11_avg_pd.append(wasp_pd)
        elif rugby.team[x] == 'sale':
            r11_avg_pd.append(sale_pd)
        elif rugby.team[x] == 'welsh':
            r11_avg_pd.append(welsh_pd)
        elif rugby.team[x] == 'exeter':
            r11_avg_pd.append(exeter_pd)
    else:
        r11_avg_pd.append(0)

rugby['r11_avg_pd'] = r11_avg_pd

for x in x_list:
    if rugby.round[x] < 12:
        if rugby.opponent[x] == 'bath':
            r11_opp_pd.append(bath_pd)
        elif rugby.opponent[x] == 'northampton':
            r11_opp_pd.append(north_pd)
        elif rugby.opponent[x] == 'gloucester':
            r11_opp_pd.append(glou_pd)
        elif rugby.opponent[x] == 'irish':
            r11_opp_pd.append(irish_pd)
        elif rugby.opponent[x] == 'harlequins':
            r11_opp_pd.append(quins_pd)
        elif rugby.opponent[x] == 'leicester':
            r11_opp_pd.append(leic_pd)
        elif rugby.opponent[x] == 'newcastle':
            r11_opp_pd.append(new_pd)
        elif rugby.opponent[x] == 'saracens':
            r11_opp_pd.append(sar_pd)
        elif rugby.opponent[x] == 'wasps':
            r11_opp_pd.append(wasp_pd)
        elif rugby.opponent[x] == 'sale':
            r11_opp_pd.append(sale_pd)
        elif rugby.opponent[x] == 'welsh':
            r11_opp_pd.append(welsh_pd)
        elif rugby.opponent[x] == 'exeter':
            r11_opp_pd.append(exeter_pd)
    else:
        r11_opp_pd.append(0)

rugby['r11_opp_pd'] = r11_opp_pd

########### creating a columen for league points based on 

league_points_single = []

for x in x_list:
    league_points = []
    if x < 239:
        if rugby.game[x] == rugby.game[x+1]:
            if rugby.tries[x] >= 4:
                league_points.append(1)
            else:
                league_points.append(0)
            if rugby.win[x] == 1:
                league_points.append(4)
            else:
                league_points.append(0)
                if -7 < rugby.points_diff[x] < (-1):
                    league_points.append(1)
                else:
                    league_points.append(0)
                    if rugby.win[x] == rugby.win[x+1]:
                        league_points.append(2)
                    else:
                        league_points.append(0)
            league_points_single.append(sum(league_points))
        elif rugby.game[x] == rugby.game[x - 1]:
            if rugby.tries[x] >= 4:
                league_points.append(1)
            else:
                league_points.append(0)
            if rugby.win[x] == 1:
                league_points.append(4)
            else:
                league_points.append(0)
                if -7 < rugby.points_diff[x] < (-1):
                    league_points.append(1)
                else:
                    league_points.append(0)
                    if rugby.win[x] == rugby.win[x - 1]:
                        league_points.append(2)
                    else:
                        league_points.append(0)
            league_points_single.append(sum(league_points))
    else:
        if rugby.tries[x] >= 4:
                league_points.append(1)
        else:
            league_points.append(0)
            if rugby.win[x] == 1:
                league_points.append(4)
            else:
                league_points.append(0)
                if -7 < rugby.points_diff[x] < (-1):
                    league_points.append(1)
                else:
                    league_points.append(0)
                    if rugby.win[x] == rugby.win[x - 1]:
                        league_points.append(2)
                    else:
                        league_points.append(0)
        league_points_single.append(sum(league_points))
    
league_points_single

rugby['league_points_awarded'] = league_points_single

###### creating columns for predicted points diff based on avg pd
###### and then using that prediction to predict win/loss for that same game

predicted_points_diff_avgPD = []
predicted_win_loss_avgPD = []

for x in x_list:
    if x == 0:
        expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x+1])
        predicted_points_diff_avgPD.append(expected)
        if expected > 0.0:
            predicted_win_loss_avgPD.append(1)
        else:
            predicted_win_loss_avgPD.append(0)
    elif 0 < x < 239:
        if rugby.game[x] == rugby.game[x+1]:
            expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x+1])
            predicted_points_diff_avgPD.append(expected)
            if expected > 0.0:
                predicted_win_loss_avgPD.append(1)
            else:
                predicted_win_loss_avgPD.append(0)
        elif rugby.game[x] == rugby.game[x-1]:
            expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x-1])
            predicted_points_diff_avgPD.append(expected)
            if expected > 0.0:
                predicted_win_loss_avgPD.append(1)
            else:
                predicted_win_loss_avgPD.append(0)
    elif x == 239:
        expected = (rugby.r11_avg_pd[x] - rugby.r11_avg_pd[x-1])
        predicted_points_diff_avgPD.append(expected)
        if expected > 0.0:
            predicted_win_loss_avgPD.append(1)
        else:
            predicted_win_loss_avgPD.append(0)

rugby['pred_PD'] = predicted_points_diff_avgPD
rugby['w_l_avgPD'] = predicted_win_loss_avgPD


########################################################################
### function for testing different home field advantages ###
########################################################################

########################################################################
#### first i'm creating a data frame with fewer columns ###########
########################################################################

tidy_rugby = rugby[['season','round','game','team','home','opponent', 
                    'w_l_avgPD','win','draw','points_diff',
                    'pred_avgPD','r11_avg_pd','league_points_awarded','r11_opp_pd']]

tidy_rugby['pred_PD'] = tidy_rugby['pred_avgPD']

tidy_rugby = tidy_rugby[['season','round','game','team','r11_avg_pd','r11_opp_pd',
                         'pred_PD','points_diff','home','opponent', 
                         'w_l_avgPD','win','draw',
                         'league_points_awarded']]

testing out a home field advantage of 5.95

predPD_home = []

for x in x_list:
    if tidy_rugby.home[x] == 1:
        predPD_home.append(tidy_rugby.r11_avg_pd[x] + 5.95 - tidy_rugby.r11_opp_pd[x])
    elif tidy_rugby.home[x] == 0:
        predPD_home.append(tidy_rugby.r11_avg_pd[x] - 5.95 - tidy_rugby.r11_opp_pd[x])

tidy_rugby['predPD_home'] = predPD_home

w_l_home = []

for x in x_list:
    if predPD_home[x] > 0.0:
        w_l_home.append(1)
    elif predPD_home[x] < 0.0:
        w_l_home.append(0)

w_l_home

tidy_rugby['w_l_home'] = w_l_home

PD_win_acc = []
home_win_acc = []

for x in x_list:
    if tidy_rugby.round[x] < 12:
        if tidy_rugby.win[x] == tidy_rugby.w_l_avgPD[x]:
            PD_win_acc.append(1)
        else:
            PD_win_acc.append(0)
        if tidy_rugby.win[x] == tidy_rugby.w_l_home[x]:
            home_win_acc.append(1)
        else:
            home_win_acc.append(0)
    else:
        PD_win_acc.append(0)
        home_win_acc.append(0)

tidy_rugby['PD_win_acc'] = PD_win_acc
tidy_rugby['home_win_acc'] = home_win_acc

PD_accuracy = sum(tidy_rugby.PD_win_acc) / 144.0
home_accuracy = sum(tidy_rugby.home_win_acc) / 144.0
PD_accuracy
home_accuracy

########################################################################
# home field advantage (5.95) improved predictions by 10% in first half
########################################################################