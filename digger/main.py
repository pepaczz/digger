import matplotlib.pyplot as plt
import pandas as pd
import itertools
from itertools import cycle
import pickle

import digger.utils as dutils
import digger.plotting as dplotting
import digger.constants as dconst
import digger.structures as dstruct
import digger.fighters as dfighters
import digger.battlefield as dbattlefield
from digger.terrain import terrains
import digger.terrain as dterrain
from datetime import datetime

from importlib import reload
reload(dplotting)

is_train = True

if __name__ == "__main__":
    # config
    models_folder = "digger_models"
    rewards_folder = "digger_rewards"
    n_battles = 3500

    dutils.maybe_make_dir(models_folder)
    dutils.maybe_make_dir(rewards_folder)

    fighters = dfighters.Fighters()
    fighters.add_fighter('arkonaut_volley', 'Gorodrin', 0)
    # fighters.add_fighter("mizzenmaster", "Gordur", 0)

    # create and reset battlefield
    battlefield = dbattlefield.Battlefield(fighters=fighters, players=[0], terrains=terrains)
    battlefield.reset(fighters)

    single_fighter = fighters.fighters[0]

    if is_train:
        single_fighter.collect_random_moves(battlefield, fighters)
    else:
        # load the previous scaler
        # NOTE JB: scaler is currently part of fighter
        # with open(f"{models_folder}/scaler.pkl", "rb") as f:
        #     scaler = pickle.load(f)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        single_fighter.epsilon = 0.01

        # load trained weights
        single_fighter.load(f"{models_folder}/dqn.ckpt")

    # clean the logs as there can be entries from initial random moves
    battlefield.action_log.clean_log()
    battlefield.battle_log.clean_log()

    # play the game n_battles times
    for e in range(n_battles):
        battlefield.play_battle(fighters, num_rounds=dconst.ROUNDS_PER_BATTLE, battle_number=e)

    # save the weights when we are done
    if is_train:
        # save the DQN
        single_fighter.save(f"{models_folder}/dqn.ckpt")

        # save the scaler
        # NOTE JB: scaler is currently part of fighter
        # with open(f"{models_folder}/scaler.pkl", "wb") as f:
        #     pickle.dump(scaler, f)

# # get battle log
battle_log = battlefield.battle_log.get_log()
start_idx = n_battles - 15
end_idx = start_idx + 8
for i in range(start_idx, end_idx):
    print(i)
    dplotting.plot_actions(fighters, battlefield, terrains, buffer=0, battle_number=i)

#
# dplotting.plot_actions(fighters, battlefield, terrains, buffer=0, battle_number=6950)
# al = battlefield.action_log.get_log(4990)





# 6950
# 6970
# 7010
# 7110
# 7790
# 7870

