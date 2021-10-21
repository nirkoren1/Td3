import gym

env = gym.make("BipedalWalkerHardcore-v3")

print(env.reward_range[0])