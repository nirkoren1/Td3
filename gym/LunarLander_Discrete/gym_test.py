import gym
import agent
import numpy as np


env = gym.make("LunarLander-v2")
ag = agent.Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape[0],
                      n_actions=4, env_high=1,
                      env_low=-1, tau=0.05, batch_size=100)  # state_size=env.observation_space.shape[0]

ag.actor.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\LunarLander_Discrete\agents\actor')
score = 0

if __name__ == '__main__':
    observation = env.reset()
    while True:
        env.render()
        action = ag.take_an_action_for_real(observation)
        action = np.argmax(action)
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            observation = env.reset()
