import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1', render_mode='human')

state, _ = env.reset()
print(f"Initial state: {state}")

state_space_size = env.observation_space.shape[0]
print(f"State space size: {state_space_size}")

action_space_size = env.action_space.n
print(f"Action space size: {action_space_size}")

action = np.random.choice(action_space_size)
print(f"Random action: {action}")
next_state, reward, terminated, _, info = env.step(action)
print(f"Next state: {next_state}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Info: {info}")

print("Playing the game...")
state, _ = env.reset()
terminated = False
reward_sum = 0

while not terminated:
    env.render()
    action = np.random.choice(action_space_size)
    next_state, reward, terminated, _, info = env.step(action)
    reward_sum += reward
    state = next_state

print(f"Total reward: {reward_sum}")

env.close()