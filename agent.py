import actor_critic
from actor_critic import ActorNet, CriticNet
from replay_buffer import ReplayBuffer
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import os


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

    def take_an_action(self, observation):
        if self.step_cntr < self.warmup:
            a = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            a = self.actor.feed_forward(state)[0]
        a += np.random.normal(scale=self.noise)
        a = tf.clip_by_value(a, self.low_limit, self.high_limit)
        self.step_cntr += 1
        return a

    def take_an_action_for_real(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        a = self.actor.feed_forward(state)[0]
        a = tf.clip_by_value(a, self.low_limit, self.high_limit)
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

        # learning of the networks
        with tf.GradientTape(persistent=True) as tape:
            # getting the target actor actions
            target_actions = self.targrt_actor.feed_forward(states_)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), - 0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions, self.low_limit, self.high_limit)

            # calculating q1 and q2 of the target nets
            q1_ = tf.squeeze(self.target_critic1.feed_forward(states_, target_actions), 1)
            q2_ = tf.squeeze(self.target_critic2.feed_forward(states_, target_actions), 1)

            # calculating q1 and q2 of the critic nets
            q1 = tf.squeeze(self.critic1.feed_forward(states, actions), 1)
            q2 = tf.squeeze(self.critic2.feed_forward(states, actions), 1)

            # getting the minimum value of the two q outputs of the critic target nets
            target_q = tf.math.minimum(q1_, q2_)
            target_q = rewards + self.gamma * (1 - dones) * target_q

            # calculate losses
            critic1_loss = tf.losses.MSE(target_q, q1)
            critic2_loss = tf.losses.MSE(target_q, q2)

        # calculate the gradients of the two critics
        critic1_gradient = tape.gradient(critic1_loss, self.critic1.trainable_weights)
        critic2_gradient = tape.gradient(critic2_loss, self.critic2.trainable_weights)

        self.critic1.optimizer.apply_gradients(zip(critic1_gradient, self.critic1.trainable_variables))
        self.critic2.optimizer.apply_gradients(zip(critic2_gradient, self.critic2.trainable_variables))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_every != 0:
            return
        with tf.GradientTape() as tape:
            actions = self.actor.feed_forward(states)
            critic1_val = self.critic1.feed_forward(states, actions)
            actor_loss = - tf.reduce_mean(critic1_val)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_nets_parameters()

    def update_nets_parameters(self):
        # update actor weights
        weights = []
        target_weights = self.targrt_actor.weights
        for idx, weight in enumerate(self.actor.weights):
            weights.append(weight * self.tau + (1 - self.tau) * target_weights[idx])
        self.targrt_actor.set_weights(weights)

        # update critic1 weights
        weights = []
        target_weights = self.target_critic1.weights
        for idx, weight in enumerate(self.critic1.weights):
            weights.append(weight * self.tau + (1 - self.tau) * target_weights[idx])
        self.target_critic1.set_weights(weights)

        # update critic2 weights
        weights = []
        target_weights = self.target_critic2.weights
        for idx, weight in enumerate(self.critic2.weights):
            weights.append(weight * self.tau + (1 - self.tau) * target_weights[idx])
        self.target_critic2.set_weights(weights)

    def save_agent(self, path, score):
        files = os.listdir(path)
        self.actor.save_weights(path)
        # self.actor.save(path + f"/{len(files) + 1}-fitness=" + str(score)[:7])
        print(f"Agent saved with {score} score")

