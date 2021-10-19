import gym
import agent


env = gym.make("BipedalWalkerHardcore-v3")
ag = agent.Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                 env_high=env.action_space.high[0], env_low=env.action_space.low[0], tau=0.05, batch_size=100)

ag.actor.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\walker_hard\agents\actor')
score = 0

if __name__ == '__main__':
    observation = env.reset()
    while True:
        env.render()
        action = ag.take_an_action_for_real(observation)
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break
    print(score)
