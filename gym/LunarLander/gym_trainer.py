import numpy as np
import gym
import sys
import os
import agent

env = gym.make("LunarLanderContinuous-v2")
ag = agent.Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape[0],
                 n_actions=4, env_high=1,
                 env_low=0, tau=0.05, batch_size=100, last_act_layer='tanh')
score_history = []


def cr_new_file():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + r'\agents'
    dirs = os.listdir(dir_path)
    if len(dirs) == 0:
        os.mkdir(dir_path + r'\run-1')
        return r'\run-1'
    else:
        os.mkdir(dir_path + rf'\run-{len(dirs) + 1}')
        return rf'\run-{len(dirs) + 1}'


if __name__ == '__main__':
    # file_destination = cr_new_file()
    loop = 0
    best_score = -1000000000000000
    while True:
        loop += 1
        sys.stdout.write(f"\r{loop}")
        sys.stdout.flush()
        observation = env.reset()
        score = 0
        while True:
            action = ag.take_an_action(observation)
            ob = observation
            observation, reward, done, info = env.step(action)
            # if reward == -100:
            #     reward = -10
            # else:
            #     reward = reward * 5
            ag.memory.save_step(ob, action, reward, observation, done)
            ag.learn()
            score += reward
            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[:-40])
        if avg_score > best_score:
            print('')
            ag.save_agent(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\LunarLander\agents\actor', score)
            best_score = avg_score
        observation = env.reset()
