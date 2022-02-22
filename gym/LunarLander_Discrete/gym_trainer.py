import numpy as np
import gym
import sys
import agent
import animate

env = gym.make("LunarLander-v2")
ag = agent.Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape[0],
                      n_actions=4, env_high=1,
                      env_low=0, tau=0.05, batch_size=100)  # state_size=env.observation_space.shape[0]
score_history = []
history_size = 40


if __name__ == '__main__':
    loop = 0
    best_score = -1000000000000000
    avg_score = best_score - 1
    while True:
        loop += 1
        observation = env.reset()
        score = 0
        while True:
            action = ag.take_an_action(observation)
            action = np.argmax(action)
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
            ag.save_agent(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\LunarLander_Discrete\agents\actor', score)
            best_score = avg_score
        sys.stdout.write(f"\rloop - {loop}  score - {score}  best - {best_score}  avg score - {avg_score}")
        sys.stdout.flush()
