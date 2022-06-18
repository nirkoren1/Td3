import gym
import numpy as np
import agent_fork
from auto_encoder import AutoEncoder
from utils import add_sensors_data_to_observation, pre_processing


env = gym.make("CarRacing-v0")
ag = agent_fork.Agent(alpha=0.001, beta=0.001, input_dims=70,
                      n_actions=env.action_space.shape[0], env_high=env.action_space.high[0],
                      env_low=env.action_space.low[0], tau=0.05, batch_size=100)

ag.actor.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\agent_fork\actor')
encoder = AutoEncoder(400, 300, 64, (28, 28))
encoder.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')
score = 0
rewards = {-8}

if __name__ == '__main__':
    observation_img = env.reset()
    while True:
        env.render()
        observation_raw = pre_processing(observation_img)
        observation = encoder.encode(observation_raw)
        observation = add_sensors_data_to_observation(observation, observation_img)
        action = ag.take_an_action_for_real(observation)
        action = np.array(action)[0]
        observation_img, reward, done, info = env.step(action)
        score += reward
        if done:
            observation = env.reset()
            print(f"score: {score}")
            score = 0
