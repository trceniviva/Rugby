# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:23:13 2015

@author: Trevor1
"""

# ( prediction - actual ) ^ 2

# calculating the predicted point difference for that team in that game
# also adding column for predicted win/loss based on the predicted point differential 

rugby = pd.read_table('aviva_fixed.csv', sep=',') # import csv sheet

predicted_points_diff = []
predicted_win_loss = []

for x in x_list:
    if x == 0:
        expected = (rugby.team_avg_pd[x] - rugby.team_avg_pd[x+1])
        predicted_points_diff.append(expected)
        if expected > 0:
            predicted_win_loss.append(1)
        else:
            predicted_win_loss.append(0)
    elif 0 < x < 239:
        if rugby.game[x] == rugby.game[x+1]:
            expected = (rugby.team_avg_pd[x] - rugby.team_avg_pd[x+1])
            predicted_points_diff.append(expected)
            if expected > 0:
                predicted_win_loss.append(1)
            else:
                predicted_win_loss.append(0)
        if rugby.game[x] == rugby.game[x-1]:
            expected = (rugby.team_avg_pd[x] - rugby.team_avg_pd[x-1])
            predicted_points_diff.append(expected)
            if expected > 0:
                predicted_win_loss.append(1)
            else:
                predicted_win_loss.append(0)
    elif x == 239:
        expected = (rugby.team_avg_pd[x] - rugby.team_avg_pd[x-1])
        predicted_points_diff.append(expected)
        if expected > 0:
            predicted_win_loss.append(1)
        else:
            predicted_win_loss.append(0)

rugby['predicted_points_diff'] = predicted_points_diff
rugby['predicted_win_loss'] = predicted_win_loss

win_accuracy = []

for x in x_list:
    if rugby.predicted_win_loss[x] == rugby.win[x]:
        win_accuracy.append(1)

sum(win_accuracy)