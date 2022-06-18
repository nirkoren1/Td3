import numpy as np
import gym
import sys
import os
import agent_fork
import animate
from auto_encoder import AutoEncoder
import cv2
from cv2 import cvtColor, resize, COLOR_BGR2GRAY
from utils import add_sensors_data_to_observation, pre_processing

env = gym.make("CarRacing-v0")
ag = agent_fork.Agent(alpha=0.001, beta=0.001, input_dims=70,
                      n_actions=env.action_space.shape[0], env_high=env.action_space.high[0],
                      env_low=env.action_space.low[0], tau=0.05, batch_size=100)
score_history = []
history_size = 20
encoder = AutoEncoder(400, 300, 64, (28, 28))
encoder.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')

if __name__ == '__main__':
    loop = 0
    best_score = -1000000000000000
    avg_score = best_score - 1
    while True:
        loop += 1
        observation_img = env.reset()
        score = 0
        observation_raw = pre_processing(observation_img)
        observation = encoder.encode(observation_raw)
        observation = add_sensors_data_to_observation(observation, observation_img)
        while True:
            action = ag.take_an_action(observation)
            ob = observation
            action = np.array(action)
            observation_img, reward, done, info = env.step(action)

            observation_raw = pre_processing(observation_img)
            observation = encoder.encode(observation_raw)
            observation = add_sensors_data_to_observation(observation, observation_img)

            observation_array = np.array(observation)[0]
            if abs(observation_array[len(observation_array) - 5]) > 0.6:
                reward -= 0.5 * abs(observation_array[len(observation_array) - 5])
            reward += observation_array[len(observation_array) - 6] * 0.05
            # reward += action[0] * 0.05

            ag.memory.save_step(ob, action, reward, observation, done)
            ag.learn()
            env.render()
            score += reward
            if done:
                break

        score_history.append(score)
        if len(score_history) >= history_size:
            avg_score = np.mean(score_history[-history_size:])
            animate.update(avg_score)
        if avg_score > best_score:
            print('')
            ag.save_agent(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\agent_fork\actor', score)
            best_score = avg_score
        sys.stdout.write(f"\rloop - {loop}  score - {score}  best - {best_score}  avg score - {avg_score}")
        sys.stdout.flush()
