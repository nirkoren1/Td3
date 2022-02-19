import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class RewardNet(keras.Model):
    def __init__(self, fc1_dims, fc2_dims):
        super(RewardNet, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.r = Dense(1, activation=None)

    def feed_forward(self, state, action, state_):
        output = self.fc1(tf.concat([state, action, state_], axis=1))
        output = self.fc2(output)
        r = self.r(output)
        return r


class StateNet(keras.Model):
    def __init__(self, l1_dims, l2_dims, state_size):
        super(StateNet, self).__init__()
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.state_size = state_size

        self.l1 = Dense(l1_dims, activation='relu')
        self.l2 = Dense(l2_dims, activation='relu')
        self.s = Dense(state_size, activation=None)

    def feed_forward(self, state, action):
        output = self.l1(tf.concat([state, action], axis=1))
        output = self.l2(output)
        s = self.s(output)
        return s