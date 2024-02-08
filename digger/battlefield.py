"""
Battlefield class holds the game state and controls the game flow.
It contains main methods for playing the game and updating the game state.
"""

import digger.utils as dutils
import digger.constants as dconst
import digger.structures as dstruct
from itertools import cycle
from shapely import distance

import digger.terrain as dterrain
import numpy as np


class Battlefield:
    def __init__(self, fighters, players, terrains):

        # players
        self.players = players
        self.players_cycle = cycle(players)

        # battle control
        self.battle_number = 0
        self.round = 0
        self.current_player = None
        self.battle_reward = 0

        # field and terrain
        self.field_min_x = 0
        self.field_min_y = 0
        self.field_max_x = dconst.FIELD_X_RANGE
        self.field_max_y = dconst.FIELD_X_RANGE
        self.terrains = terrains

        # fighters
        self.fighter_ids = fighters.fighter_ids
        self.fighter_positions = {}
        self.fighters_cycles = {}

        # action space - each dict item for one fighter
        # self.action_spaces = {}
        self.remaining_actions = None

        # logs
        self.action_log = dstruct.ActionLog()
        self.move_paths = {}
        self.battle_log = dstruct.BattleLog()

        # target
        self.target_pos = (None, None)

    def reset(self, fighters):
        """Reset the battlefield before each battle"""
        self.battle_number += 1
        self.round = 0

        # set target position
        target_x = dconst.TARGET_X
        target_y = dconst.TARGET_Y
        self.target_pos = (target_x, target_y)

        # reset fighters positions and rewards
        for idx, fighter in enumerate(fighters.get_list()):

            # set position of each fighter
            new_x_pos = dconst.FIELD_X_RANGE / 2 + idx * 2
            self.fighter_positions[fighter.fighter_id] = (new_x_pos, 3)

            # reset fighter
            fighter.reset()

        # set fighters_cycles
        fighters_df = fighters.get_fighters_df()
        for player in self.players:
            fighters_player_now = fighters_df.loc[
                fighters_df.player_id == player, "fighter_id"
            ].to_list()
            self.fighters_cycles[player] = cycle(fighters_player_now)

    def get_state(self):
        """Get state of the game. When changing state size, change also STATE_SIZE in constants.py!
        Hardcoded currently to single fighter - its position and angle and distance to target.
        In future this should retrieve more complex states including:
        - all fighters positions
        - obstacles
        - other fighters' stats
        - the above possibly in form of some sort of vision as input for CNN
        """
        # get angle and distance to target
        angle, distance_to_tgt = dutils.get_angle_and_distance(self.fighter_positions[0],
                                                               self.target_pos)

        state = np.array([self.fighter_positions[0][0], self.fighter_positions[0][1],
                          angle, distance_to_tgt])

        # other possible states - comment out!
        # state = np.array([self.fighter_positions[0][0], self.fighter_positions[0][1],
        #                   self.target_pos[0], self.target_pos[1],
        #                   angle, distance_to_tgt])
        # state = np.array([angle, distance_to_tgt])

        return state

    def play_battle(self, fighters, num_rounds, is_train=True, battle_number=0):
        """Play the whole battle
        In simple setup this means 4 rounds and each fighter has two actions per round
        """
        # print('#### Starting battle ####')
        self.battle_number = battle_number
        self.reset(fighters)

        # iterate over rounds
        for round_number in range(num_rounds):
            self.play_round(fighters, is_train)
            if len(fighters.get_not_done_fighters()) == 0:
                break

        # get information for logs and print
        single_fighter = fighters.fighters[0]
        reward = np.round(single_fighter.battle_reward, 3)
        n_rounds = self.round
        epsilon = np.round(single_fighter.epsilon, 3)

        # ## DEBUG PART - COMMENT OUT!
        # self.battle_number
        # al = self.action_log.get_log(self.battle_number)
        # all = self.action_log.get_log()
        # # sum reward
        # r1 = al['reward'].sum()
        # r2 = single_fighter.battle_reward
        # if abs(r1 - r2) > 0.2:
        #     print(f'Rewards do not match: {r1} vs {r2}')
        #     print(f"xxx")
        # if r1 > dconst.REW_REACHED_TARGET:
        #     print('Rewards too high')
        #     print(f"xxx")
        # ## END DEBUG PART

        # append to battle log
        self.battle_log.append_to_battle_log(
            self.battle_number, n_rounds, len(fighters.fighter_ids),
            reward, self.target_pos
        )

        if battle_number % dconst.PRINT_EVERY_NTH_BATTLE == 0:
            print(f'Battle: {battle_number} || Rew: {reward} || Eps: {epsilon} || Rnds: {n_rounds}')

    def play_round(self, fighters, is_train=True):
        """Play single round of the game, i.e. each fighter has two actions in basic setup.
        """
        # initialize round and get current state
        self.round += 1
        self.current_player = dconst.PLAYER_START
        state = self.get_state()

        # reset fighters' remaining actions
        self.remaining_actions = fighters.get_reset_remaining_actions()
        end_of_round = False

        # loop over actions within round
        while not end_of_round:
            # placeholder for some sort of active_fighter selection
            # next fighter is selected from fighters_cycles
            # current_player is already set in set_next_player()
            active_fighter_id = next(self.fighters_cycles[self.current_player])
            active_fighter = fighters.fighters[active_fighter_id]
            # print(f'Activating fighter {active_fighter_id}')

            # step
            action = active_fighter.act(state)
            next_state, reward, done, info = self.step(action, active_fighter)
            next_state = active_fighter.scaler.transform([next_state])

            # update replay memory and train model
            if is_train:
                active_fighter.update_replay_memory(state, action, reward, next_state, done)
                active_fighter.replay(dconst.BATCH_SIZE)

            # set next player and check whether round is over
            self.set_next_current_player()
            state = next_state
            if self.current_player is None:
                end_of_round = True

        return None

    def set_next_current_player(self):
        """Search and set next player for the round.
        Players are cycled in fixed order, but only those with remaining actions are considered.
        """
        # get remaining actions by player
        players_remn_actions = self.remaining_actions.groupby("player_id").sum()["remaining_actions"]

        # set next player with nonzero remaining actions
        self.current_player = None
        if sum(players_remn_actions) > 0:
            candidate_found = False

            # cycle over players until candidate is found
            while not candidate_found:
                next_cand_player = next(self.players_cycle)
                if players_remn_actions[next_cand_player] > 0:
                    self.current_player = next_cand_player
                    candidate_found = True

        return None

    def step(self, action, active_fighter, lower_remain_actions=True):
        """Perform single step in the game.
        Hardcoded as move action, but in future it can be other types of actions (shooting, combat, ability...)
        Lowering or zeroing remaining actions is a bit cumbersome now.
        """
        active_fighter_id = active_fighter.fighter_id
        done = False

        # perform move action
        log_index = self.perform_move_action(action, active_fighter)

        # decrease remaining actions (not done during initial random moves collection)
        if lower_remain_actions:
            self.lower_fighter_remaining_actions(active_fighter_id)

        # set fighter as done if no remaining actions and last round
        remaining_actions = self.get_fighter_remaining_actions(active_fighter_id)
        if remaining_actions == 0 and self.round == dconst.ROUNDS_PER_BATTLE:
            active_fighter.is_done = True
            done = True

        # retrieve new position and calculate distance to target
        pos = self.fighter_positions[active_fighter_id]
        pos_x, pos_y = pos
        distance_to_tgt = distance(dutils.point_to_shapely(self.target_pos), dutils.point_to_shapely(pos))

        # standard reward - closer to target better
        reward = dconst.REW_DIST_MULT * distance_to_tgt

        # fighter ends on the edge
        if (pos_x <= self.field_min_x) or (pos_y <= self.field_min_y) or \
                (pos_x >= self.field_max_x) or (pos_y >= self.field_max_y):
            reward = dconst.REW_FIELD_EDGE
            active_fighter.is_done = True
            done = True

        # fighter reaches target
        if distance_to_tgt <= 3:
            reward = dconst.REW_REACHED_TARGET
            print('**** target reached! ****')
            active_fighter.is_done = True
            done = True

        # set remaining actions to zero if fighter is done
        if done:
            self.zero_fighter_remaining_actions(active_fighter_id)

        # log the state and reward
        self.action_log.write_to_action_log(log_index, state=self.get_state(), reward=np.round(reward, 3))

        info = {'current_position': self.fighter_positions[active_fighter_id],
                'distance_to_tgt': distance_to_tgt}

        # add to total reward
        active_fighter.battle_reward += reward

        return self.get_state(), reward, done, info

    def perform_move_action(self, action, active_fighter):
        """Perform move action for a single fighter.
        Avoids obstacles and logs the action.
        to implement: treat other fighters as obstacles as well
        """
        # get fighter and move definition
        f_id = active_fighter.fighter_id
        move_def = active_fighter.standard_moves_def[action]
        start_point = self.fighter_positions[f_id]
        end_point = dutils.move_by_angle_and_distance(
            start_point, move_def[0], move_def[1] * active_fighter.stats['Move'])

        # get terrain and expand it by fighter base radius to avoid collision
        obstacles = dterrain.get_terrains_union(active_fighter.base / 2)

        # get restricted move
        end_point_restricted, path_restricted = dutils.get_restricted_move(
            start_point, end_point, obstacles, active_fighter.stats['Move'], get_path=True)
        self.fighter_positions[f_id] = end_point_restricted

        # log the action
        action_order = self.action_log.get_fighter_round_actions(self.round, f_id, self.battle_number).shape[0]
        self.action_log.append_to_action_log(
            self.current_player, self.battle_number, self.round, f_id, action_order, 'move', action,
            move_def, start_point, self.fighter_positions[f_id])

        # save the restricted move path
        log_index = self.action_log.action_log.index[-1]
        self.move_paths[log_index] = path_restricted

        return log_index

    def add_fighter(self):
        """Register new fighter in the battlefield and return its id
        Fighter position is stored in the battlefield object
        """
        new_id = int(len(self.fighter_ids))
        self.fighter_ids.append(new_id)
        self.fighter_positions[new_id] = (None, None)
        return new_id

    def set_fighter_position(self, fighter_id, point):
        """Set position of the fighter in the battlefield"""
        self.fighter_positions[fighter_id] = point

    def get_fighter_remaining_actions(self, fighter_id):
        """Return remaining actions for given fighter"""
        return self.remaining_actions.loc[
            self.remaining_actions.fighter_id == fighter_id,
            'remaining_actions'
        ].values[0]

    def lower_fighter_remaining_actions(self, fighter_id):
        """Lower remaining actions for given fighter by 1"""
        self.remaining_actions.loc[
            self.remaining_actions.fighter_id == fighter_id,
            'remaining_actions'
        ] -= 1

    def zero_fighter_remaining_actions(self, fighter_id):
        """Set remaining actions for given fighter to 0
        Currently used in situation when fighter reaches
        """
        self.remaining_actions.loc[
            self.remaining_actions.fighter_id == fighter_id,
            'remaining_actions'
        ] = 0

