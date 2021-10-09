import numpy as np


class ReplayBuffer:
    def __init__(self, input_dim, action_dim, mem_size):
        self.cntr = 0
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.states = np.zeros(input_dim, mem_size)
        self.actions = np.zeros(action_dim, mem_size)
        self.rewards = np.zeros(mem_size)
        self.states_ = np.zeros(input_dim, mem_size)

    def save_step(self, state, action, ):
        pass