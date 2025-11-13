import time

import gymnasium as gym
import numpy as np

from exports import load_numpy_array

bins = np.array((40, 40))


def discrete_values(state, env):
    bin_size = (env.observation_space.high - env.observation_space.low) / bins
    state = (state - env.observation_space.low) / bin_size
    state = np.floor(state).astype(int)
    return np.clip(state, 0, bins - 1)


def train():
    env = gym.make('MountainCar-v0')

    q_table = np.zeros((*bins, env.action_space.n))
    # print("Q Table Shape", q_table.shape)
    # print("Discrete values for state", state, discrete_values(state))
    #
    # print(q_table[discrete_values(state), ].shape)
    # print(q_table[*discrete_values(state)].shape)

    alpha = 0.4
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.9998
    min_epsilon = 0.01
    max_episodes = 100_000

    def choose_action(state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        return np.argmax(q_table[*state])

    for episode in range(max_episodes):
        state, _ = env.reset()
        stop = False
        total_reward = 0
        state = discrete_values(state, env)

        while not stop:
            action = choose_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = discrete_values(next_state, env)
            total_reward += reward

            next_best_action_qvalue = np.max(q_table[*next_state])
            td_target = reward + gamma * next_best_action_qvalue
            td_error = td_target - q_table[*state, action]
            q_table[*state, action] += (alpha * td_error)
            stop = terminated or truncated
            state = next_state

        epsilon = max(epsilon_decay * epsilon, min_epsilon)
        print("Episode", episode, "Total reward", total_reward, "Epsilon", epsilon)
    env.close()
    return q_table


def test(q_table):
    env = gym.make('MountainCar-v0', render_mode='human')
    for i in range(30):
        stop = False
        state, _ = env.reset()
        state = discrete_values(state, env)
        total_reward = 0
        while not stop:
            env.render()
            action = np.argmax(q_table[*state])
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = discrete_values(next_state, env)
            state = next_state
            stop = terminated or truncated
            time.sleep(0.01)
            total_reward += reward
        print("Episode", i, "Total reward", total_reward)


q_table_file_path = './trained_tables/MountainCar-v2.npy'

# trained_table = train()
# export_numpy_array(trained_table, q_table_file_path)

trained_table = load_numpy_array(q_table_file_path)

test(q_table=trained_table)
