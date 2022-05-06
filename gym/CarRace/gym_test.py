import gym
import agent_2D
import cv2
from cv2 import resize, cvtColor, COLOR_BGR2GRAY
import numpy as np
import keyboard
import itertools
from utils import get_sensors_data_from_images, get_sensors_pic


env = gym.make("CarRacing-v0")
# ag = agent_2D.Agent(alpha=0.001, beta=0.001, input_dims=(28, 28),
#                     n_actions=env.action_space.shape[0], env_high=env.action_space.high[0],
#                     env_low=env.action_space.low[0], tau=0.05,
#                     batch_size=100, latent_dim=64,
#                     auto_encoder_path=r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')

# ag.actor.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\agents\actor')
# ag.auto_encoder.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')
score = 0




if __name__ == '__main__':
    observation = env.reset()
    player_action = [0, 0, 0]
    acceleration = 0.1
    steering_acceleration = 0.2
    while True:
        env.render()
        sensors_pics = get_sensors_pic(observation)
        print(list(get_sensors_data_from_images(sensors_pics[0], sensors_pics[1], sensors_pics[2], sensors_pics[3],
                                           sensors_pics[4], sensors_pics[5])))
        player_action[0] *= acceleration
        player_action[1] *= acceleration
        player_action[2] *= acceleration
        if keyboard.is_pressed('left'):
            player_action[0] -= steering_acceleration
        elif keyboard.is_pressed('right'):
            player_action[0] += steering_acceleration
        if keyboard.is_pressed('up'):
            player_action[1] += acceleration
        if keyboard.is_pressed('space'):
            player_action[2] += acceleration
        # observation_raw = pre_processing(observation)
        # observation = ag.auto_encoder.encode(observation_raw)
        # action = ag.take_an_action_for_real(observation)
        # action = (2 / (1 + np.e ** (-2 * (action ** 5)))) - 1
        observation, reward, done, info = env.step(player_action)
        score += reward
        if done:
            # observation = env.reset()
            # print(f"score: {score}")
            score = 0
