import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler


DATA_FILE = 'aapl_msi_sbux.csv'

def get_data(data_file):
    """Loads data from csv file

    :return:
    """
    return pd.read_csv(data_file).values


def get_scaler(env):

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class MultiStockEnv:

    def __init__(self, data, initial_investment=20000):

        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3**self.n_stock)

        self.action_list = list(
            map(list, itertools.product([0, 1, 2], repeat=self.n_stock))
        )
        self.state_dim = self.n_stock * 2 + 1
        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def step(self, action):
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        # get the value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store current value in output dictionary
        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        action_vec = self.action_list[action]

        sell_index = []
        buy_index = []

        # collect sell and buy indices
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        if sell_index:
            for sell_idx in sell_index:
                self.cash_in_hand += self.stock_owned[sell_idx] * self.stock_price[sell_idx]
                self.stock_owned[sell_idx] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for buy_idx in buy_index:
                    if self.cash_in_hand >= self.stock_price[buy_idx]:
                        self.cash_in_hand -= self.stock_price[buy_idx]
                        self.stock_owned[buy_idx] += 1
                    else:
                        can_buy = False


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.max_size, self.ptr + 1)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s=self.obs1_buf[idxs],
            s2=self.obs2_buf[idxs],
            a=self.acts_buf[idxs],
            r=self.rews_buf[idxs],
            d=self.done_buf[idxs],
        )


class MLP(nn.Module):
    def __init__(self, n_inputs, n_action, n_hidden_layers=1, hidden_dim=32):
        super().__init__()

        M = n_inputs

        self.layers = []

        for _ in range(n_hidden_layers):
            layer = nn.Linear(M, hidden_dim)
            self.layers.append(layer)
            self.layers.append(nn.ReLU())
            M = hidden_dim

        # final layer
        self.layers.append(nn.Linear(M, n_action))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.layers(X)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


def predict(model, np_states):
    with torch.no_grad():
        inputs = torch.from_numpy(np_states.astype(np.float32))
        output = model(inputs)
        return output.numpy()


def train_one_step(model, criterion, optimizer, inputs, targets):

    inputs = torch.from_numpy(inputs.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))

    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # backward pass
    loss.backward()
    optimizer.step()


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = MLP(state_size, action_size)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            act_value = predict(self.model, state)
            return np.argmax(act_value[0])

    def replay(self, batch_size=32):

        if self.memory.size < batch_size:
            return

        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        target = rewards + (1 - done) * self.gamma * np.amax(
            predict(self.model, next_states), axis=1
        )

        # ????
        target_full = predict(self.model, states)
        target_full[np.arange(batch_size), actions] = target

        # run training step
        train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, scaler, is_train, batch_size):
    # reset env, buffer?

    done = False
    state = env.reset()
    state = scaler.transform([state])

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])

        if is_train == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)
        state = next_state

    return info['cur_val']



models_folder = "rl_trader_models"
rewards_folder = "rl_trader_rewards"
num_episodes = 50
batch_size = 32
initial_investment = 20000

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode", type=str, required=True, help='either "train" or "test"'
)
# args = parser.parse_args()
args = parser.parse_args(['-m', 'test'])   # train test

maybe_make_dir(models_folder)
maybe_make_dir(rewards_folder)

data = get_data(DATA_FILE)
n_timesteps, n_stocks = data.shape

n_train = n_timesteps // 2

train_data = data[:n_train]
test_data = data[n_train:]

env = MultiStockEnv(train_data, initial_investment)
state_size = env.state_dim
action_size = len(env.action_space)

agent = DQNAgent(state_size, action_size)
scaler = get_scaler(env)

portfolio_value = []

if args.mode == "test":
    # then load the previous scaler
    with open(f"{models_folder}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)
    agent.epsilon = 0.01
    agent.load(f"{models_folder}/dqn.ckpt")

for e in range(num_episodes):
    t0 = datetime.now()
    val = play_one_episode(agent, env, scaler, args.mode, batch_size)
    dt = datetime.now() - t0
    print(
        f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}"
    )
    portfolio_value.append(val)  # append episode end portfolio value

# save the weights when we are done
if args.mode == "train":
    # save the DQN
    agent.save(f"{models_folder}/dqn.ckpt")

    # save the scaler
    with open(f"{models_folder}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# save portfolio value for each episode
np.save(f"{rewards_folder}/{args.mode}.npy", portfolio_value)