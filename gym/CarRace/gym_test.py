from time import sleep

import gym
import numpy as np

import agent_2D
import keyboard
from utils import add_sensors_data_to_observation, pre_processing, get_sensors_data_from_images, get_sensors_pic


env = gym.make("CarRacing-v0")
ag = agent_2D.Agent(alpha=0.001, beta=0.001, input_dims=(28, 28),
                    n_actions=env.action_space.shape[0], env_high=env.action_space.high[0],
                    env_low=env.action_space.low[0], tau=0.05,
                    batch_size=100, latent_dim=64, sensors_size=6,
                    auto_encoder_path=r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')

ag.actor.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\agents\actor')
ag.auto_encoder.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')
score = 0


if __name__ == '__main__':
    observation_img = env.reset()
    # player_action = [0, 0, 0]
    # acceleration = 0.1
    # steering_acceleration = 0.2
    while True:
        env.render()
        # player_action[0] *= acceleration
        # player_action[1] *= acceleration
        # player_action[2] *= acceleration
        # if keyboard.is_pressed('left'):
        #     player_action[0] -= steering_acceleration
        # elif keyboard.is_pressed('right'):
        #     player_action[0] += steering_acceleration
        # if keyboard.is_pressed('up'):
        #     player_action[1] += acceleration
        # if keyboard.is_pressed('space'):
        #     player_action[2] += acceleration
        # sensors_pic = get_sensors_pic(observation_img)
        # print(get_sensors_data_from_images(sensors_pic[0], sensors_pic[1], sensors_pic[2], sensors_pic[3], sensors_pic[4], sensors_pic[5]))
        observation_raw = pre_processing(observation_img)
        observation = ag.auto_encoder.encode(observation_raw)
        observation = add_sensors_data_to_observation(observation, observation_img)
        action = ag.take_an_action_for_real(observation)
        action = np.array(action)
        observation_img, reward, done, info = env.step(action)
        score += reward
        if done:
            observation = env.reset()
            print(f"score: {score}")
            score = 0
