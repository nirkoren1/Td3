import numpy as np
import gym
import sys
import agent_2D
import animate
from utils import add_sensors_data_to_observation, pre_processing
import record_helper

recorder = record_helper.RecordHelper()
env = gym.make("CarRacing-v0")
ag = agent_2D.Agent(alpha=0.001, beta=0.001, input_dims=(28, 28),
                    n_actions=env.action_space.shape[0], env_high=env.action_space.high[0],
                    env_low=env.action_space.low[0], tau=0.05,
                    batch_size=100, latent_dim=64, sensors_size=6,
                    auto_encoder_path=r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')
score_history = []
history_size = 25
auto_encoder_exist = False
if auto_encoder_exist:
    ag.auto_encoder.load_weights(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\auto_encoder\weights')
    ag.auto_encoder.is_saved = True


if __name__ == '__main__':
    loop = 0
    best_score = -1000000000000000
    avg_score = best_score - 1
    while True:
        loop += 1
        observation_img = env.reset()
        score = 0
        observation_raw = pre_processing(observation_img)
        observation = ag.auto_encoder.encode(observation_raw)
        observation = add_sensors_data_to_observation(observation, observation_img)
        while True:
            recorder.print_time(loop)
            action = ag.take_an_action(observation)
            ob = observation
            action = np.array(action)
            observation_img, reward, done, info = env.step(action)

            observation_raw = pre_processing(observation_img)
            observation = ag.auto_encoder.encode(observation_raw)
            observation = add_sensors_data_to_observation(observation, observation_img)

            observation_array = np.array(observation)[0]
            if abs(observation_array[len(observation_array) - 5]) > 0.6:
                reward -= 0.5 * abs(observation_array[len(observation_array) - 5])
            reward += observation_array[len(observation_array) - 6] * 0.05

            ag.memory.save_step(ob, action, reward, observation, done, observation_raw)
            ag.learn()
            # env.render()
            score += reward
            if done:
                break

        score_history.append(score)
        if len(score_history) >= history_size:
            avg_score = np.mean(score_history[-history_size:])
            animate.update(avg_score, r"C:\Users\Nirkoren\PycharmProjects\Td3\animate_data\car_race")
        if avg_score > best_score:
            # print('')
            # ag.save_agent(r'C:\Users\Nirkoren\PycharmProjects\Td3\gym\CarRace\agents\actor', score)
            best_score = avg_score
        # sys.stdout.write(f"\rloop - {loop}  score - {score}  best - {best_score}  avg score - {avg_score}")
        # sys.stdout.flush()
        recorder.render_loop(loop, env, ag, observation)
