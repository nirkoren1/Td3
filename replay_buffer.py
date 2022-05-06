import numpy as np


class ReplayBuffer:
    def __init__(self, input_dim, action_dim, mem_size):
        self.cntr = 0
        self.mem_size = mem_size
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.states = np.zeros((self.mem_size, input_dim))
        self.actions = np.zeros((self.mem_size, action_dim))
        self.rewards = np.zeros(self.mem_size)
        self.states_ = np.zeros((self.mem_size, input_dim))
        self.dones = np.zeros(self.mem_size, dtype=np.bool)

    def save_step(self, state, action, reward, state_, done):
        index = self.cntr % self.mem_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = state_
        self.dones[index] = done
        self.cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        states_ = self.states_[batch]
        dones = self.dones[batch]
        return states, actions, rewards, states_, dones


class ReplayBuffer2D:
    def __init__(self, encode_dim, action_dim, mem_size, input_shape):
        self.cntr = 0
        self.mem_size = mem_size
        self.encode_dim = encode_dim
        self.action_dim = action_dim
        self.state_raw = np.zeros((self.mem_size, input_shape[0], input_shape[1]), dtype=np.float32)
        self.states = np.zeros((self.mem_size, encode_dim))
        self.actions = np.zeros((self.mem_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.mem_size)
        self.states_ = np.zeros((self.mem_size, encode_dim))
        self.dones = np.zeros(self.mem_size, dtype=np.bool)

    def save_step(self, state, action, reward, state_, done, state_raw):
        index = self.cntr % self.mem_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = state_
        self.dones[index] = done
        self.state_raw[index] = state_raw
        self.cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        states_ = self.states_[batch]
        dones = self.dones[batch]
        raws = self.state_raw[batch]
        return states, actions, rewards, states_, dones, raws
