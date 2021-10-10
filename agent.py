import actor_critic
from actor_critic import ActorNet, CriticNet
from replay_buffer import ReplayBuffer
import tensorflow as tf
import tensorflow.keras as keras
from keras.optimizers import Adam
import numpy as np


class Agent:
    def __init__(self, alpha, beta, input_dims, n_actions, env_high, env_low, tau, gamma=0.99, update_actor_every=2,
                 max_size=1_000_000, layer1_size=400, layer2_size=300, batch_size=300, noise=0.1, warmup=1_000):
        self.n_actions = n_actions
        self.high_limit = env_high
        self.low_limit = env_low
        self.tau = tau
        self.gamma = gamma
        self.update_actor_every = update_actor_every
        self.memory = ReplayBuffer(input_dims, n_actions, max_size)
        self.batch_size = batch_size
        self.warmup = warmup
        self.noise = noise
        self.learn_step_cntr = 0
        self.step_cntr = 0

        self.actor = ActorNet(layer1_size, layer2_size, n_actions)
        self.critic1 = CriticNet(layer1_size, layer2_size)
        self.critic2 = CriticNet(layer1_size, layer2_size)

        self.targrt_actor = ActorNet(layer1_size, layer2_size, n_actions)
        self.target_critic1 = CriticNet(layer1_size, layer2_size)
        self.target_critic2 = CriticNet(layer1_size, layer2_size)

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.critic2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        self.targrt_actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.target_critic1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_critic2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

    def take_an_action(self, observasion):
        if self.step_cntr < self.warmup:
            a = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observasion], dtype=tf.float32)
            a = self.actor.feed_forward(state)[0]
        a += np.random.normal(scale=self.noise)
        a = tf.clip_by_value(a, self.low_limit, self.high_limit)
        self.step_cntr += 1
        return a

    def remember(self, state, action, reward, state_, done):
        self.memory.save_step(state, action, reward, state_, done)

    def learn(self):
        if self.memory.cntr < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
