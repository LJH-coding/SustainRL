import argparse
import os
import time

import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
from gymnasium.wrappers import FlattenObservation, RescaleAction
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import TD7


def make_env(seed=0):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
        env = FlattenObservation(env)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        return env
    return _init


def make_eval_env(seed=1):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
        env = FlattenObservation(env)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        return env
    return _init


def train_online(RL_agent, env, eval_env, cfg):
    evals = []
    start_time = time.time()
    allow_train = False

    state, _ = env.reset()
    ep_finished = False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    # Create models directory
    models_dir = f"{cfg.log_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    for t in range(int(cfg.timesteps+1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, cfg)
        
        # Save model periodically
        if t > 0 and t % cfg.checkpoint_freq == 0:
            model_path = f"{models_dir}/td7_checkpoint_{t}.pth"
            RL_agent.save(model_path)
        
        if allow_train:
            action = RL_agent.select_action(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, ep_finished, truncated, _ = env.step(action)
        ep_finished = ep_finished or truncated
        
        ep_total_reward += reward
        ep_timesteps += 1

        done = float(ep_finished)
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train and not cfg.use_checkpoints:
            RL_agent.train()

        if ep_finished: 
            print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")

            if allow_train:
                if cfg.use_checkpoints:
                    RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if t >= cfg.timesteps_before_training:
                allow_train = True

            state, _ = env.reset()
            ep_finished = False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1 

    # Save final model
    final_model_path = f"{models_dir}/td7_final.pth"
    RL_agent.save(final_model_path)
    
    # Save policy only (smaller file for evaluation)
    policy_path = f"{models_dir}/td7_policy.pth"
    RL_agent.save_policy_only(policy_path)
    
    print(f"Training completed! Final model saved to {final_model_path}")
    print(f"Policy saved to {policy_path}")


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, cfg):
    if t % cfg.eval_freq == 0:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(cfg.eval_episodes_during_training)
        for ep in range(cfg.eval_episodes_during_training):
            state, _ = eval_env.reset()
            done = False
            while not done:
                action = RL_agent.select_action(state, cfg.use_checkpoints, use_exploration=False)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward[ep] += reward

        print(f"Average total reward over {cfg.eval_episodes_during_training} episodes: {total_reward.mean():.3f}")
        
        print("---------------------------------------")

        evals.append(total_reward)
        
        # Create results directory if it doesn't exist
        results_dir = f"{cfg.log_dir}/results"
        os.makedirs(results_dir, exist_ok=True)
        np.save(f"{results_dir}/TD7_EVCharging_{cfg.seed}", evals)


def evaluate_policy(RL_agent, eval_env, n_episodes=100, use_checkpoints=True):
    """
    Evaluate a trained policy
    """
    print(f"Evaluating policy for {n_episodes} episodes...")
    
    total_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = RL_agent.select_action(state, use_checkpoints, use_exploration=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: Reward = {episode_reward:.3f}")
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("---------------------------------------")
    print(f"Evaluation Results:")
    print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Min Reward: {np.min(total_rewards):.3f}")
    print(f"Max Reward: {np.max(total_rewards):.3f}")
    print("---------------------------------------")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'all_rewards': total_rewards,
        'all_lengths': episode_lengths
    }

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    # Prepare directories
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create results directory
    results_dir = f"{log_dir}/results"
    os.makedirs(results_dir, exist_ok=True)

    env = make_env()()
    eval_env = make_eval_env()()

    print("---------------------------------------")
    print(f"Algorithm: TD7, Env: EVCharging-v0")
    print("---------------------------------------")
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}")

    # Create TD7 hyperparameters
    hp = TD7.Hyperparameters(**cfg.TD7)

    RL_agent = TD7.Agent(state_dim, action_dim, max_action, hp=hp)

    print("Starting training...")
    train_online(RL_agent, env, eval_env, cfg)
    print("Training completed!")

if __name__ == "__main__":
    main()