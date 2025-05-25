import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
from gymnasium.wrappers import FlattenObservation
from tqdm import tqdm
import numpy as np

gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
#env = FlattenObservation(env)

print(env.observation_space)
print(env.action_space)
print(env.spec.max_episode_steps)

ls = []
money = []

for _ in tqdm(range(100)):
    total_return = 0
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        total_return += reward
    ls.append(total_return)
    money.append(info['reward_breakdown']['profit'])

print(np.mean(ls), np.std(ls))
print(np.mean(money), np.std(money))
