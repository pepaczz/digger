"""
This module contains classes for logging individual fighters' actions and battle logs.
"""
import pandas as pd
import numpy as np

class ActionLog:
    def __init__(self):
        self.action_log = pd.DataFrame(
            columns=[
                'player_id',
                'battle_number',
                'round',
                'fighter_id',
                'action_order',
                'action_type',
                'action',
                'action_details',
                'position_start',
                'position_end',
                'state',
                'reward'
            ]
        )

    def append_to_action_log(
        self, player_id, battle_number, round, fighter_id, action_order, action_type,
            action, action_details, position_start, position_end, state=None, reward=None
    ):
        """Append single action to action log"""
        self.action_log.loc[len(self.action_log), :] = [
            player_id,
            battle_number,
            round,
            fighter_id,
            action_order,
            action_type,
            action,
            action_details,
            tuple(np.round(position_start, 3)) if position_start is not None else None,
            tuple(np.round(position_end, 3)) if position_end is not None else None,
            state,
            reward
        ]
        return self.action_log

    def write_to_action_log(self, log_index, state=None, reward=None):
        if state is not None:
            self.action_log.loc[log_index, ['state']] = [state]
        if reward is not None:
            self.action_log.loc[log_index, ['reward']] = reward

    def get_log(self, battle_number=None):
        """Get action log"""
        if battle_number is not None:
            return self.action_log[self.action_log['battle_number'] == battle_number]
        else:
            return self.action_log

    def clean_log(self):
        self.action_log = self.action_log.iloc[0:0]

    def get_fighter_round_actions(self, round, fighter_id, battle_number):
        """Get all actions in given round for given fighter"""
        return self.action_log[(self.action_log['round'] == round) &
                               (self.action_log['fighter_id'] == fighter_id) &
                               (self.action_log['battle_number'] == battle_number)]


class BattleLog:
    def __init__(self):
        self.battle_log = pd.DataFrame(
            columns=[
                'battle_number',
                'n_rounds',
                'n_fighters',
                'fighter_reward',
                'target_position'
            ]
        )

    def append_to_battle_log(
        self, battle_number, n_rounds, n_fighters, fighter_reward, target_position
    ):
        """Append single action to action log"""
        self.battle_log.loc[len(self.battle_log), :] = [
            battle_number,
            n_rounds,
            n_fighters,
            fighter_reward,
            tuple(np.round(target_position, 3))
        ]
        return self.battle_log

    def get_log(self, battle_number=None):
        """Get action log"""
        if battle_number is not None:
            return self.battle_log[self.battle_log['battle_number'] == battle_number]
        else:
            return self.battle_log

    def clean_log(self):
        self.battle_log = self.battle_log.iloc[0:0]
