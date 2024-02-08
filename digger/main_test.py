"""
FOR TESTING PURPOSES ONLY. RUN main.py INSTEAD
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from itertools import cycle

import digger.utils as dutils
import digger.plotting as dplotting
import digger.constants as dconst
import digger.structures as dstruct
import digger.fighters as dfighters
import digger.battlefield as dbattlefield
from digger.terrain import terrains
import digger.terrain as dterrain

fighters = dfighters.Fighters()
# fighters.add_fighter('mizzenmaster', 'Gordur', 0)
fighters.add_fighter('arkonaut_volley', 'Bruddumri', 0)
# fighters.add_fighter('arkonaut_volley', 'Gorodrin', 0)
fighters.get_living_fighters()
fighter = fighters.fighters[0]

# create and reset battlefield
battlefield = dbattlefield.Battlefield(fighters=fighters, players=[0], terrains=terrains)
battlefield.reset(fighters)

# get some attributes
battlefield.get_state()

# fit scaler
single_fighter = fighters.fighters[0]
_ = single_fighter.collect_random_moves(battlefield, fighters)

# play battle
battlefield.play_battle(fighters, num_rounds=4)




import math
fighter_pos = (0, 0)
target_pos = (-2, -2)
dif = (target_pos[0] - fighter_pos[0], target_pos[1] - fighter_pos[1])

# np.rad2deg(math.atan(dif[1] / dif[0]))
# angle = np.rad2deg(np.arctan(dif[1] / dif[0]))
angle = np.rad2deg(np.arctan2(dif[1], dif[0]))
distance = np.sqrt(dif[0]**2 + dif[1]**2)
angle

dutils.move_by_angle_and_distance((0, 0), angle, distance)



random_moves = fighter.collect_random_moves(battlefield, fighters)
random_moves = fighter.collect_random_moves(battlefield, fighters)
random_moves = fighter.collect_random_moves(battlefield, fighters)
len(random_moves)
random_moves[0]


# testing movement
print(battlefield.fighter_positions)
battlefield.perform_move_action(0, fighter)
battlefield.perform_move_action(0, fighter)
print(battlefield.fighter_positions)

# battlefield.perform_move_action(0, fighter)
print('Starting round')
print(battlefield.fighter_positions)
battlefield.play_round(fighters)
battlefield.play_round(fighters)
battlefield.play_round(fighters)
battlefield.play_round(fighters)
print(battlefield.fighter_positions)

# plot
dplotting.plot_actions(fighters, battlefield, terrains, buffer=0)

# show log
log = battlefield.action_log.get_log()
log


#############

# move_angles = [30, 90, 150, 210, 270, 330]
# move_ratios = [1, 2/3, 1/3]
# standard_moves_def = list(itertools.product(move_angles, move_ratios))
# # generate_move_points
# max_move = 10
# start_point = (11, 3)
# obstacles = terrains
# end_points_final, shortest_paths = dutils.get_restricted_std_moves(standard_moves_def, max_move, start_point, obstacles)
# dplotting.plot_terrains_and_lines(obstacles, shortest_paths)








