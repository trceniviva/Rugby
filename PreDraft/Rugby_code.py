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
#### Now i'm going to create a data frame with fewer columns ###########
########################################################################

tidy_rugby = rugby[['season','round','game','team','home','opponent', 
                    'w_l_avgPD','win','draw','points_diff',
                    'pred_avgPD','r11_avg_pd','league_points_awarded','r11_opp_pd']]

tidy_rugby['pred_PD'] = tidy_rugby['pred_avgPD']

tidy_rugby = tidy_rugby[['season','round','game','team','r11_avg_pd','r11_opp_pd',
                         'pred_PD','points_diff','home','opponent', 
                         'w_l_avgPD','win','draw',
                         'league_points_awarded']]
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

########################################################################
## I'd like to create a column that tracks average point              ##
## differential over the season. Or rather, that shows each team's    ##
## current average point differential going into that game.           ##
## This should allow me to then calculate expected PD more            ##
## effectively, and in real time for additional matches.              ##
########################################################################

########################################################################
## First, I have created a function that gives me a list of average   ##
## point differentials throughout the course of the season.           ##
########################################################################

def team_pd_change(a):
    this_team = a
    y_round = list(range(2,22))
    team_pd_list = []
    for y in y_round:
        team_pd_list.append(tidy_rugby[tidy_rugby.team == this_team][tidy_rugby.round < y].points_diff.mean())
    print team_pd_list
    
team_pd_change('bath')

########################################################################
## First, I have created a function that gives me a list of average   ##
## point differentials throughout the course of the season.           ##
########################################################################

teams = teams[0:12]

for team in teams:    
    team_dict = {: team_pd_change(team)}
    


R12 = tidy_rugby[tidy_rugby.round < 13]

R12[['team','r11_avg_pd','r11_opp_pd','pred_PD','points_diff']].head(12)
R12[['team','w_l_avgPD','win']].head(12)

### error producer funcion ###

def error_producer(words):
bath_pred = words[0]
north_pred = words[1]
glou_pred = words[2]
irish_pred = words[3]
quins_pred = words[4]
leic_pred = words[5]
new_pred = words[6]
sar_pred = words[7]
wasp_pred = words[8]
sale_pred = words[9]
welsh_pred = words[10]
exeter_pred = words[11]

predictions = []

### adding round 1 predicted differences

predictions.append(north_pred-glou_pred)
predictions.append(irish_pred-quins_pred)
predictions.append(leic_pred-new_pred)
predictions.append(sar_pred-wasp_pred)
predictions.append(sale_pred-bath_pred)
predictions.append(welsh_pred-exeter_pred)

### adding round 2 predicted differences

predictions.append(quins_pred-sar_pred)
predictions.append(exeter_pred-leic_pred)
predictions.append(glou_pred-sale_pred)
predictions.append(bath_pred-welsh_pred)
predictions.append(new_pred-irish_pred)
predictions.append(wasp_pred-north_pred)

### adding round 3 predicted differences

predictions.append(glou_pred-exeter_pred)
predictions.append(bath_pred-leic_pred)
predictions.append(irish_pred-sar_pred)
predictions.append(quins_pred-wasp_pred)
predictions.append(sale_pred-welsh_pred)
predictions.append(new_pred-north_pred)

### adding round 4 predicted differences

predictions.append(welsh_pred-glou_pred)
predictions.append(north_pred-bath_pred)
predictions.append(sar_pred-sale_pred)
predictions.append(leic_pred-irish_pred)
predictions.append(wasp_pred-new_pred)
predictions.append(exeter_pred-quins_pred)

### adding round 5 predicted differences

predictions.append(bath_pred-sar_pred)
predictions.append(glou_pred-leic_pred)
predictions.append(irish_pred-north_pred)
predictions.append(quins_pred-welsh_pred)
predictions.append(new_pred-exeter_pred)
predictions.append(sale_pred-wasp_pred)

### adding round 6 predicted differences

predictions.append(leic_pred-quins_pred)
predictions.append(north_pred-sale_pred)
predictions.append(sar_pred-glou_pred)
predictions.append(exeter_pred-irish_pred)
predictions.append(welsh_pred-new_pred)
predictions.append(wasp_pred-bath_pred)

### adding round 7 predicted differences

predictions.append(north_pred-exeter_pred)
predictions.append(glou_pred-quins_pred)
predictions.append(bath_pred-new_pred)
predictions.append(sale_pred-irish_pred)
predictions.append(wasp_pred-welsh_pred)
predictions.append(leic_pred-sar_pred)

### adding round 8 predicted differences

predictions.append(new_pred-glou_pred)
predictions.append(quins_pred-sale_pred)
predictions.append(irish_pred-bath_pred)
predictions.append(exeter_pred-wasp_pred)
predictions.append(welsh_pred-leic_pred)
predictions.append(sar_pred-north_pred)

### add round 9 predicted differences

predictions.append(bath_pred-quins_pred)
predictions.append(exeter_pred-sar_pred)
predictions.append(leic_pred-wasp_pred)
predictions.append(new_pred-sale_pred)
predictions.append(welsh_pred-north_pred)
predictions.append(irish_pred-glou_pred)

### round 10 predicted differences

predictions.append(sale_pred-exeter_pred)
predictions.append(glou_pred-bath_pred)
predictions.append(north_pred-leic_pred)
predictions.append(sar_pred-welsh_pred)
predictions.append(quins_pred-new_pred)
predictions.append(wasp_pred-irish_pred)

### round 11 predicted differences

predictions.append(irish_pred-welsh_pred)
predictions.append(quins_pred-north_pred)
predictions.append(new_pred-sar_pred)
predictions.append(sale_pred-leic_pred)
predictions.append(bath_pred-exeter_pred)
predictions.append(glou_pred-wasp_pred)    

predictions    
    
    x_len = list(range(0,len(predictions)))
    
    psq_less_actual = []
    
    for x in x_len:
        psq_less_actual.append((predictions[x] - actual_diff[x])**2)
        
    print sum(psq_less_actual)

########################################################################
########################################################################

actual_diff = []

for x in x_list:
        if rugby.round[x] < 12:
            if x < 239:
                if rugby.game[x] == rugby.game[x+1]:
                    if rugby.home[x] == 1:
                        actual_diff.append(rugby.points_diff[x])
                    if rugby.home[x] == 0:
                        actual_diff.append(rugby.points_diff[x+1])
                        
            if x > 0:
                if rugby.game[x] == rugby.game[x-1]:
                    if rugby.home[x] == 1:
                        actual_diff.append(rugby.points_diff[x])
                    if rugby.home[x] == 0:
                        actual_diff.append(rugby.points_diff[x - 1])
        

round_12_diff = []

for x in x_list:
    if rugby.home[x] == 1:
        if rugby.round[x] == 12:
            round_12_diff.append(rugby.points_diff[x])
            
### using average points per game to predict games

bath_ppg = 27.10
north_ppg = 28.05
glou_ppg = 24.25
irish_ppg = 19.30
quins_ppg = 19.70
leic_ppg = 20.10
new_ppg = 20.50
sar_ppg = 28.80
wasp_ppg = 30.40
sale_ppg = 22.20
welsh_ppg = 10.30
exeter_ppg = 27.35

error_avg_ppg = [bath_ppg,north_ppg,glou_ppg,
                 irish_ppg,quins_ppg,leic_ppg,
                 new_ppg,sar_ppg,wasp_ppg,
                 sale_ppg,welsh_ppg,exeter_ppg]

error_producer(error_avg_ppg)

### using average point differential to predict games

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

words = error_avg_pd

error_producer(error_avg_pd)

########################################################################
########################################################################


###### model for using avg point differential #####

pred_error = []

for x in x_list:
    if x < 239:
        if rugby.game[x] == rugby.game[x+1]:
            pred_error.append((((rugby.team_avg_pd[x]) - (rugby.team_avg_pd[x+1]) - ((rugby.points[x])-(rugby.points[x+1])))**2))

sum(pred_error)

###### generic model for when I have set predictions #######

x_list = list(range(0,240))

predictions = []

for x in x_list:
    if teams[x] == 'bath':
        predictions.append(bath_pred)
    elif teams[x] == 'northampton':
        predictions.append(north_pred)
    elif teams[x] == 'gloucester':
        predictions.append(glou_pred)
    elif teams[x] == 'irish':
        predictions.append(irish_pred)
    elif teams[x] == 'harlequins':
        predictions.append(quins_pred)
    elif teams[x] == 'leicester':
        predictions.append(leic_pred)
    elif teams[x] == 'newcastle':
        predictions.append(new_pred)
    elif teams[x] == 'saracens':
        predictions.append(sar_pred)
    elif teams[x] == 'wasps':
        predictions.append(wasp_pred)
    elif teams[x] == 'sale':
        predictions.append(sale_pred)
    elif teams[x] == 'welsh':
        predictions.append(welsh_pred)
    else:
        predictions.append(exeter_pred)

pred_error = []

for x in x_list:
    if x < 239:
        if game[x] == game[x+1]:
            pred_error.append((((5.85 + predictions[x]) - predictions[x+1]) - (points[x]-points[x+1]))**2)

sum(pred_error)


########################################################################

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

team_df = [bath, northampton, gloucester, irish, harlequins, leicester,
           newcastle, saracens, wasps, sale, welsh, exeter]

from pandas import concat

rugby = concat(team_df)

rugby.set_index(list(range(0,240)), drop=True)