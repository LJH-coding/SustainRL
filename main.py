import argparse
import os
import time

import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch

import M2TD7


def make_env(seed=0):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
        env = FlattenObservation(env)
        return env
    return _init


def make_eval_env(seed=1):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
        env = FlattenObservation(env)
        return env
    return _init


def train_online(RL_agent, env, eval_env, args):
    evals = []
    start_time = time.time()
    allow_train = False

    state, _ = env.reset()
    ep_finished = False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    for t in range(int(args.max_timesteps+1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
        
        if allow_train:
            action = RL_agent.select_action(np.array(state))
            omega, _, _ = RL_agent.get_omega()  # Get omega parameter
        else:
            action = env.action_space.sample()
            omega = np.random.uniform(
                RL_agent.omega_min, RL_agent.omega_max
            )  # Random omega during exploration

        next_state, reward, ep_finished, truncated, _ = env.step(action)
        ep_finished = ep_finished or truncated
        
        ep_total_reward += reward
        ep_timesteps += 1

        done = float(ep_finished)
        RL_agent.replay_buffer.add(state, action, next_state, reward, done, omega)

        state = next_state

        if allow_train and not args.use_checkpoints:
            RL_agent.train()

        if ep_finished: 
            print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")

            if allow_train:
                RL_agent.set_current_episode_len(ep_timesteps)
                if args.use_checkpoints:
                    RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if t >= args.timesteps_before_training:
                allow_train = True

            state, _ = env.reset()
            ep_finished = False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1 



def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args):
    if t % args.eval_freq == 0:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(args.eval_eps)
        for ep in range(args.eval_eps):
            state, _ = eval_env.reset()
            done = False
            while not done:
                action = RL_agent.select_action(state, args.use_checkpoints, use_exploration=False)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward[ep] += reward

        print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
        
        print("---------------------------------------")

        evals.append(total_reward)
        np.save(f"./results/{args.file_name}", evals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--use_checkpoints', default=True, action=argparse.BooleanOptionalAction)
    # Evaluation
    parser.add_argument("--timesteps_before_training", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e7, type=int)
    # M2TD7 specific
    parser.add_argument("--omega_dim", default=2, type=int)
    parser.add_argument("--omega_min", default=[-1.0, -1.0], nargs='+', type=float)
    parser.add_argument("--omega_max", default=[1.0, 1.0], nargs='+', type=float)
    parser.add_argument("--hatomega_num", default=5, type=int)
    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = f"M2TD7_EVCharging_{args.seed}"

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Create environments using your specific setup
    env = make_env(seed=args.seed)()
    eval_env = make_eval_env(seed=args.seed+100)()

    print("---------------------------------------")
    print(f"Algorithm: M2TD7, Env: EVCharging-v0, Seed: {args.seed}")
    print("---------------------------------------")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}")

    # Create M2TD7 hyperparameters
    hp = M2TD7.Hyperparameters(
        omega_dim=args.omega_dim,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        hatomega_num=args.hatomega_num
    )

    RL_agent = M2TD7.M2TD7Agent(state_dim, action_dim, max_action, hp=hp)

    print("Starting training...")
    train_online(RL_agent, env, eval_env, args)
    print("Training completed!")