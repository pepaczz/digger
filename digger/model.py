import torch
import torch.nn as nn
import numpy as np

# The experience replay memory
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
        self.size = min(self.size + 1, self.max_size)

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
    def __init__(self, n_inputs, n_action, n_hidden_layers, hidden_dim):
        super(MLP, self).__init__()

        M = n_inputs
        self.layers = []
        for _ in range(n_hidden_layers):
            layer = nn.Linear(M, hidden_dim)
            M = hidden_dim
            self.layers.append(layer)
            self.layers.append(nn.ReLU())

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
        # print("output:", output)
        return output.numpy()


def train_one_step(model, criterion, optimizer, inputs, targets):
    # convert to tensors
    inputs = torch.from_numpy(inputs.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))

    # zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    loss.backward()
    optimizer.step()
