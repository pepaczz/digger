"""
Contains classes for fighters.
Fighter class is used for each individual fighter.
Fighters class is used to manage the collection of fighters.
"""
import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import digger.utils as dutils
import digger.constants as dconst
from digger.model import MLP, ReplayBuffer, train_one_step, predict


def get_fighter_stats(fighter_class='mizzenmaster', filename='stats_ko.xlsx'):
    stats = pd.read_excel(os.path.join(dconst.STATS_FOLDER, filename))
    fighter_stats = stats.loc[stats.FighterClass == fighter_class, :].to_dict(orient='records')[0]
    return fighter_stats


class Fighter:
    def __init__(self, fighter_class, name, player_id):
        self.player_id = player_id
        self.fighter_id = None
        self.fighter_class = fighter_class
        self.name = name
        self.stats = get_fighter_stats(self.fighter_class)
        self.base = np.round(dutils.mm_to_inches(self.stats['BaseMM']), 2)

        # standard moves
        self.move_angles = [30, 90, 150, 210, 270, 330]
        self.move_ratios = [1, 1/2]  # [1, 2/3, 1/3]
        self.standard_moves_def = list(itertools.product(self.move_angles, self.move_ratios))

        # action space
        self.action_space = np.arange(len(self.standard_moves_def))
        self.action_size = len(self.action_space)

        # model and memory
        self.memory = ReplayBuffer(dconst.STATE_SIZE, self.action_size, size=dconst.BUFFER_SIZE)
        self.gamma = 0.85  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99993
        self.model = MLP(dconst.STATE_SIZE, self.action_size,
                         n_hidden_layers=dconst.N_HIDDEN_LAYERS, hidden_dim=dconst.HIDDEN_DIM)
        self.scaler = None

        # loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # dynamic attributes (reset on each battle)
        self.battle_reward = None
        self.is_done = None
        self.current_wounds = None

        self.reset()

    def reset(self):
        """Resets dynamic attributes for the fighter. Called at the start of each battle."""
        self.current_wounds = self.stats['Wnd']
        self.battle_reward = 0
        self.is_done = False

        # although normally constant, stats might change due to some abilities
        self.stats = get_fighter_stats(self.fighter_class)

    def update_replay_memory(self, state, action, reward, next_state, done):
        """Update replay memory with new data"""
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        # TODO: see why there was originally np.argmax(act_values[0])?
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = predict(self.model, state)
        return np.argmax(act_values)  # returns action

    def collect_random_moves(self, battlefield, fighters, update_memory=True, fit_scaler=True):
        """Collect random moves to populate the replay buffer and fit the scaler
        Note: records are biased towards initial moves.
            There are fewer records for moves that are further in the game (target, combat...)

        To think: in future probably move this to battlefield as the scaler should be common to all fighters?
            On the other hand the replay memory is specific to each fighter
        """
        states = []
        not_enough_records = True
        while not_enough_records:

            battlefield.reset(fighters)
            done = False

            while not done:
                battlefield.remaining_actions = fighters.get_reset_remaining_actions()
                state = battlefield.get_state()
                action = np.random.choice(self.action_space)
                next_state, reward, done, info = battlefield.step(
                    action=action, active_fighter=self, lower_remain_actions=False)
                states.append(next_state)
                if update_memory:
                    self.update_replay_memory(state, action, reward, next_state, done)

            if len(states) > dconst.SCALER_INIT_ITERATIONS:
                not_enough_records = False

        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(states)

        return states

    def replay(self, batch_size):
        """Train the model using the replay buffer"""
        # first check if replay buffer contains enough data
        if self.memory.size < batch_size:
            return

        # sample a batch of data from the replay memory
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch["s"]
        actions = minibatch["a"]
        rewards = minibatch["r"]
        next_states = minibatch["s2"]
        done = minibatch["d"]

        # Calculate the target: Q(s',a)
        target = rewards + (1 - done) * self.gamma * np.amax(
            predict(self.model, next_states), axis=1
        )

        # With the PyTorch API, it is simplest to have the target be the
        # same shape as the predictions.
        # However, we only need to update the network for the actions
        # which were actually taken.
        # We can accomplish this by setting the target to be equal to
        # the prediction for all values.
        # Then, only change the targets for the actions taken.
        # Q(s,a)
        target_full = predict(self.model, states)
        target_full[np.arange(batch_size), actions] = target

        # Run one training step
        train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load model weights from file"""
        self.model.load_weights(name)

    def save(self, name):
        """Save model weights to file"""
        self.model.save_weights(name)


class Fighters:
    def __init__(self):
        self.fighters = {}
        self.fighter_ids = []

    def add_fighter(self, fighter_class, name, player_id):
        """Add a new fighter to the collection of fighters"""
        new_fighter = Fighter(fighter_class, name, player_id)

        # generate and assign fighter_id
        new_id = int(len(self.fighter_ids))
        self.fighter_ids.append(new_id)
        new_fighter.fighter_id = new_id

        # add fighter to the collection of fighters
        self.fighters[new_id] = new_fighter

        return new_id

    def get_list(self):
        """Return list containing all fighters objects"""
        return self.fighters.values()

    def get_living_fighters(self):
        """Return list of fighters with current wounds > 0"""
        return [f.fighter_id for f in self.get_list() if f.current_wounds > 0]

    def get_not_done_fighters(self):
        """Return list of fighters with is_done = False"""
        return [f.fighter_id for f in self.get_list() if f.is_done is False]

    def get_fighters_df(self, fighter_ids=None):
        """Return fighters info for selected fighter_ids"""
        if fighter_ids is None:
            fighter_ids = self.fighter_ids

        df = pd.DataFrame(columns=['fighter_id', 'player_id', 'fighter_class', 'name', 'current_wounds'])
        for f in self.get_list():
            if f.fighter_id in fighter_ids:
                new_row = {
                    'fighter_id': f.fighter_id,
                    'player_id': f.player_id,
                    'fighter_class': f.fighter_class,
                    'name': f.name,
                    'current_wounds': f.current_wounds}
                df = pd.concat([df, pd.DataFrame([new_row])], sort=False)
        return df

    def get_reset_remaining_actions(self):
        """Creates dataframe for remaining actions for all fighters
        Resets to 2 actions for each fighter
        """
        res = self.get_fighters_df(self.get_living_fighters())
        res['remaining_actions'] = 2
        return res.loc[:, ['fighter_id', 'player_id', 'remaining_actions']]