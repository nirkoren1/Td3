import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os


class CriticNet(keras.Model):
    def __init__(self, fc1_dims, fc2_dims):
        super(CriticNet, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def feed_forward(self, state, action):
        output = self.fc1(tf.concat([state, action], axis=1))
        output = self.fc2(output)
        q = self.q(output)
        return q


class ActorNet(keras.Model):
    def __init__(self, l1_dims, l2_dims, n_actions, last_act_layer):
        super(ActorNet, self).__init__()
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.n_actions = n_actions

        self.l1 = Dense(l1_dims, activation='relu')
        self.l2 = Dense(l2_dims, activation='relu')
        self.a = Dense(n_actions, activation="tanh")

    def feed_forward(self, state):
        output = self.l1(state)
        output = self.l2(output)
        a = self.a(output)
        return a