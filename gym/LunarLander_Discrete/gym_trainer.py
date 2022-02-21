import numpy as np
import gym
import sys
import os
import agent_fork
import animate

env = gym.make("LunarLanderContinuous-v2")
ag = agent_fork.Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape[0],
                      n_actions=env.action_space.shape[0], env_high=env.action_space.high[0],
                      env_low=env.action_space.low[0], tau=0.05, batch_size=100,
                      state_size=env.observation_space.shape[0])  # state_size=env.observation_space.shape[0]
score_history = []
history_size = 40


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
    avg_score = best_score - 1
    while True:
        loop += 1
        observation = env.reset()
        score = 0
        while True:
            action = ag.take_an_action(observation)
            ob = observation
            observation, reward, done, info = env.step(action)
            ag.memory.save_step(ob, action, reward, observation, done)
            ag.learn()
            score += reward
            if done:
                break

        score_history.append(score)
        if len(score_history) >= history_size:
            avg_score = np.mean(score_history[-history_size:])
            animate.update(avg_score)
        if avg_score > best_score:
            print('')
            ag.save_agent(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\LunarLander\agent_fork2\actor', score)
            best_score = avg_score
        sys.stdout.write(f"\rloop - {loop}  score - {score}  best - {best_score}  avg score - {avg_score}")
        sys.stdout.flush()
