import time

import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v3')
q_table = np.zeros((env.observation_space.n, env.action_space.n))
# print(q_table)
# print(q_table.shape)
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
max_episodes = 10000
gamma = 0.9
alpha = 0.9


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[state,])


for episode in range(max_episodes):
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        next_best_qvalue = np.max(q_table[next_state,])
        td_target = reward + gamma * next_best_qvalue
        td_error = td_target - q_table[state, action]
        q_table[state, action] += alpha * td_error

        done = terminated or truncated
        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print("Episode", episode, "Total reward", total_reward, "Epsilon", epsilon)

env.close()

# testing

env = gym.make('Taxi-v3', render_mode='human')
for episode in range(10):
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action = np.argmax(q_table[state,])
        next_state, reward, terminated, truncated, infor = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.3)

    print("Test Episode", episode, "Reward", total_reward)
