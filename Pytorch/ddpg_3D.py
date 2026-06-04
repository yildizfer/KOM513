"""
PyTorch DDPG Implementation for 3D Continuum Robot Control

DDPG (Deep Deterministic Policy Gradient) is an actor-critic algorithm that:
1. Actor network: learns deterministic policy π(s) -> a
2. Critic network: learns value function Q(s, a) -> scalar value

Key differences from TensorFlow/Keras version:
- More flexible model definition
- Better for custom architectures and debugging
- Automatic differentiation with PyTorch's autograd

3D Changes:
- State size: 6 (x, y, z, goal_x, goal_y, goal_z)
- Action size: 6 (κ̇₁, κ̇₂, κ̇₃, φ̇₁, φ̇₂, φ̇₃)
"""

import sys
sys.path.append('./RL-based-Control-of-a-Soft-Continuum-Robot/Reinforcement Learning')

import torch
print("Device:", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from env import continuumEnv
import time
import math
import os
import yaml

# Load configuration
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, "config.yaml")
with open(file_path, "r") as file:
    config = yaml.safe_load(file)

start_time = time.time()

TRAIN = True  # Set to True to train, False to evaluate

# Initialize environment
env = continuumEnv()

# Create agent with 3D state and action spaces
# State size: 16 (x, y, z, goal_x, goal_y, goal_z, kappa_1..3, phi_1..3, error_x, error_y, error_z, distance_to_goal)
# Action size: 6 (3 curvature rates + 3 bending angle rates)
agent = Agent(state_size=16, action_size=6, random_seed=10)

def ddpg(n_episodes=6000, max_t=1000, print_every=100):
    """
    Deep Deterministic Policy Gradient Training Loop

    Args:
        n_episodes: Number of episodes to train
        max_t: Maximum timesteps per episode
        print_every: Print stats every N episodes

    Returns:
        scores: List of episodic rewards
    """
    global scores
    global avg_reward_list

    scores_deque = deque(maxlen=print_every)
    scores = []
    avg_reward_list = []
    counter = 0
    distances_at_reset = []
    actions_magnitude = []

    for i_episode in range(1, n_episodes+1):
        # Reset environment: random start and goal positions
        state = env.reset()

        # Reset agent's noise process
        agent.reset()

        score = 0

        # Print episode information
        if i_episode % print_every == 0:
            print('\n')
            # Print first 3 state elements (current position x, y, z)
            print("Initial Position is", env.state[0:3])
            print("===============================================================")
            # Print last 3 state elements (goal position x, y, z)
            print("Target Position is", env.state[3:6])
            print("===============================================================")
            print("Initial Kappas are ", [env.kappa1, env.kappa2, env.kappa3])
            print("===============================================================")
            print("Initial Phis are ", [env.phi1, env.phi2, env.phi3])
            print("===============================================================")
            time.sleep(0.5)

        episode_distances = []
        episode_actions = []
        episode_rewards = []

        # Run episode
        for t in range(max_t):
            # Actor selects action based on current policy + exploration noise
            # Decay exploration noise scale dynamically over 80% of training episodes down to 0.01
            decay_episodes = int(n_episodes * 0.8)
            noise_scale = max(0.01, 1.0 - i_episode / decay_episodes)
            action = agent.act(state, add_noise=True, noise_scale=noise_scale)
            episode_actions.append(np.linalg.norm(action))

            # Execute action in environment
            # Available reward functions:
            # - 'step_minus_euclidean_square': -distance^2 (smooth)
            # - 'step_minus_weighted_euclidean': -0.7*distance
            # - 'step_error_comparison': discrete rewards
            # - 'step_distance_based': shaped rewards
            next_state, reward, done, _ = env.step(
                action,
                reward_function=config['reward']['function']
            )
            env.render_calculate()

            # Track distance to goal (physical meters)
            dist = env.state[15]
            episode_distances.append(dist)

            # Track rewards for diagnostics
            episode_rewards.append(reward)

            # Agent processes the experience:
            # 1. Stores in replay buffer
            # 2. Learns from random batch if buffer is full
            agent.step(state, action, reward, next_state, done)

            # Move to next state
            state = next_state

            # Accumulate episode reward
            score += reward

            # Episode ends when goal reached
            if done:
                counter += 1
                break

        # Store episode score
        scores_deque.append(score)
        scores.append(score)

        # Track diagnostics
        if episode_distances:
            distances_at_reset.append(episode_distances[0])
            actions_magnitude.append(np.mean(episode_actions))

        # Compute moving average over recent episodes (helps smooth learning signal)
        avg_reward_list.append(np.mean(scores[-100:]))

        # Print progress with reward diagnostics
        reward_range = f"[{np.min(episode_rewards):.3f}, {np.max(episode_rewards):.3f}]" if episode_rewards else "N/A"
        print('\rEpisode {}\tAverage Score: {:.2f}\tReward Range: {}\tAvg Init Dist: {:.4f}m\tAvg Action: {:.4f}'.format(
            i_episode, np.mean(scores_deque),
            reward_range,
            np.mean(distances_at_reset[-print_every:]) if distances_at_reset else 0,
            np.mean(actions_magnitude[-print_every:]) if actions_magnitude else 0), end="")

        # Save checkpoints every episode (for recovery)
        torch.save(agent.actor_local.state_dict(), './RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/experiment/checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), './RL-based-Control-of-a-Soft-Continuum-Robot/Pytorch/experiment/checkpoint_critic.pth')

    print('\n')
    print(f'{counter} times robot reached the target point in total {n_episodes} episodes')
    print(f'Success rate: {counter/n_episodes*100:.1f}%')
    end_time = time.time() - start_time
    print('Total Workspace Overshoots: ', env.overshoot0)
    print('Total Goal Overshoots: ', env.overshoot1)
    print(f'Total Elapsed Time is {int(end_time)/60} minutes')
    print(f'Average initial-target distance: {np.mean(distances_at_reset):.4f}m (range: {np.min(distances_at_reset):.4f}m to {np.max(distances_at_reset):.4f}m)')
    print(f'Average action magnitude: {np.mean(actions_magnitude):.4f}')

    return scores

# =====================================================================
#                     Main Execution
# =====================================================================

if TRAIN:
    # Run training
    scores = ddpg()

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    # Plot moving average
    ax1.plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Moving Average (100 episodes)")
    ax1.grid()

    # Plot episodic scores
    ax2.plot(np.arange(1, len(scores)+1), scores, alpha=0.3)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episodic Reward')
    ax2.set_title("Per-Episode Rewards")
    ax2.grid()

    plt.tight_layout()
    plt.savefig('results.png')
    # plt.show()

    # Save training results
    with open('scores.pickle', 'wb') as f:
        pickle.dump(scores, f)
    with open('avg_reward_list.pickle', 'wb') as f:
        pickle.dump(avg_reward_list, f)

else:
    # Evaluation mode: load trained weights and run policy
    print("Loading pre-trained weights...")
    # Uncomment to load and evaluate
    # agent.actor_local.load_state_dict(torch.load('experiment/checkpoint_actor.pth'))
    # agent.critic_local.load_state_dict(torch.load('experiment/checkpoint_critic.pth'))
    pass
