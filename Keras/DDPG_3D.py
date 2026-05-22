"""
DDPG (Deep Deterministic Policy Gradient) Algorithm for 3D Continuum Robot Control

This script implements DDPG in Keras/TensorFlow for training a policy to control
a 3-segment soft continuum robot to reach target positions in 3D space.

Key changes for 3D (from 2D):
- State space: 6D (x, y, z, goal_x, goal_y, goal_z) instead of 4D
- Action space: 6D (kappa_dot_1/2/3, phi_dot_1/2/3) instead of 3D
- Networks automatically scale with dimension changes
- Reward calculated in 3D space using Euclidean distance
"""

import sys
sys.path.append('../')
sys.path.append('../Reinforcement Learning')
sys.path.append('../Tests')

import os
import yaml
import tensorflow as tf
tf.config.set_soft_device_placement(True)
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
from tensorflow.keras import layers
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from env import continuumEnv

# Load configuration file
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, "config.yaml")
with open(file_path, "r") as file:
    config = yaml.safe_load(file)

env = continuumEnv()

# Dynamically determine state and action dimensions from environment
# For 3D: observation_space.shape[0] = 3 (one 3D point in sphere space)
# Multiplied by 2 for concatenated state: [current_position, goal_position]
# Result: 3 * 2 = 6D observation space [x, y, z, goal_x, goal_y, goal_z]
num_states = env.observation_space.shape[0] * 2
print("Size of State Space ->  {}".format(num_states))

# For 3D: action_space.shape[0] = 6
# 3 curvature rates (kappa_dot_1, kappa_dot_2, kappa_dot_3) +
# 3 bending angle rates (phi_dot_1, phi_dot_2, phi_dot_3)
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
start_time = time.time()

# =====================================================================
#                     Ornstein-Uhlenbeck Noise
# =====================================================================
class OUActionNoise:
    """
    Ornstein-Uhlenbeck process for correlated action noise exploration.

    This noise process is important for DDPG as it provides temporally correlated
    exploration that helps the algorithm discover distant rewards more effectively
    than uncorrelated noise (like Gaussian noise).

    The noise follows: x_t = x_{t-1} + theta*(mean - x_{t-1})*dt + sigma*sqrt(dt)*N(0,1)
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """
        Initialize the OU noise process.

        Args:
            mean: Target mean value of the noise (typically 0)
            std_deviation: Standard deviation (exploration intensity)
            theta: Mean reversion rate (0.15 is typical, higher = faster reversion)
            dt: Time step for discretization
            x_initial: Initial state (None uses zeros)
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """Generate next noise sample using OU process update equation."""
        # OU process differential equation discretized
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        """Reset noise state to initial value."""
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# =====================================================================
#                     Experience Replay Buffer
# =====================================================================
class Buffer:
    """
    Fixed-size circular experience replay buffer for DDPG.

    Stores transitions (s, a, r, s') and samples random batches for training.
    This breaks temporal correlations in training data and improves stability.
    """
    def __init__(self, buffer_capacity=100000, batch_size=64):
        """
        Initialize replay buffer.

        Args:
            buffer_capacity: Maximum number of experiences to store
            batch_size: Size of mini-batches for training
        """
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0  # Tracks total experiences recorded

        # Pre-allocate arrays for efficiency
        # Each array stores one component of the transition tuple
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        """
        Store a new experience in the buffer using circular indexing.

        Args:
            obs_tuple: (state, action, reward, next_state) transition
        """
        # Circular index: wrap around when buffer is full
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]  # Remove extra dimension
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function  # JIT compile for speed: TensorFlow builds static computation graph
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        """
        Update actor and critic networks using a batch of experiences.

        DDPG Algorithm:
        1. Critic learns Q-function: minimize MSE(Q(s,a), r + gamma*Q_target(s',a'))
        2. Actor learns policy: maximize E[Q(s, policy(s))]
        3. Soft update target networks: θ_target = tau*θ_local + (1-tau)*θ_target

        Args:
            state_batch: Batch of states (size: batch_size x num_states)
            action_batch: Batch of actions (size: batch_size x num_actions)
            reward_batch: Batch of rewards (size: batch_size x 1)
            next_state_batch: Batch of next states (size: batch_size x num_states)
        """

        # =====================================================
        # CRITIC UPDATE: Estimate Q-value function
        # =====================================================
        with tf.GradientTape() as tape:
            # Get target action from target actor network
            # Actor produces action for next state
            target_actions = target_actor(next_state_batch, training=TRAIN)

            # Compute target Q-value using target networks (for stability)
            # Q_target(s', a') comes from "frozen" target critic
            # This prevents instability from moving targets
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=TRAIN
            )

            # Compute predicted Q-value using local (learning) critic
            # Q_local(s, a) is our estimate of state-action value
            critic_value = critic_model([state_batch, action_batch], training=TRAIN)

            # Critic loss: MSE between predicted and target Q-values
            # Minimizing this makes the critic learn accurate value estimates
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # Backpropagation: compute gradients of critic loss w.r.t. critic weights
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        # Apply gradients: update critic weights to reduce loss
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        # =====================================================
        # ACTOR UPDATE: Maximize Q-value for current state
        # =====================================================
        with tf.GradientTape() as tape:
            # Get actions from local (learning) actor for current states
            actions = actor_model(state_batch, training=TRAIN)

            # Compute Q-values for the actions produced by actor
            # Higher Q-value means better action for the given state
            critic_value = critic_model([state_batch, actions], training=TRAIN)

            # Actor loss: negative mean Q-value (we want to MAXIMIZE Q)
            # This is the policy gradient: dJ/dθ_actor ∝ dQ/dθ_actor
            actor_loss = -tf.math.reduce_mean(critic_value)

        # Backpropagation: compute gradients of actor loss w.r.t. actor weights
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        # Apply gradients: update actor to maximize Q-value (move in gradient direction)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(self):
        """
        Sample random batch from buffer and update networks.

        This is called after each environment step. Learning only starts
        once buffer has enough samples (batch_size).
        """
        # Sample a mini-batch of experiences uniformly at random
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert numpy arrays to TensorFlow tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Perform network updates using the sampled batch
        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# =====================================================================
#                     Target Network Updates
# =====================================================================
@tf.function
def update_target(target_weights, weights, tau):
    """
    Soft update target network weights.

    Target networks are slowly updated to stabilize learning.
    θ_target = tau * θ_local + (1 - tau) * θ_target

    where tau << 1 (typically 5e-3), so target changes slowly.
    This prevents the moving target problem and improves stability.

    Args:
        target_weights: Weights to update
        weights: Source weights to copy from
        tau: Soft update coefficient (typically 0.005)
    """
    for (a, b) in zip(target_weights, weights):
        # Weighted average: mostly keep old target weights, mix in small portion of local
        a.assign(b * tau + a * (1 - tau))

# =====================================================================
#                     Actor Network
# =====================================================================
def get_actor():
    """
    Build the Actor (Policy) network.

    Maps states to actions: state (6D) -> action (6D)
    The actor learns a deterministic policy π(s) that maximizes expected Q-value.

    Architecture:
    - Input: 6D state [x, y, z, goal_x, goal_y, goal_z]
    - Hidden layers: 512 -> 256 -> 128 ReLU units
    - Output: 6D action [κ̇₁, κ̇₂, κ̇₃, φ̇₁, φ̇₂, φ̇₃] bounded to [-1, 1] via tanh

    Note: Output is in normalized space [-1, 1] because action_space.high[0] = 1.0
    """
    # Initialize final layer weights uniformly from small interval
    # This helps stabilize initial learning
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # Input layer: takes full 6D state
    inputs = layers.Input(shape=(num_states,))

    # Hidden layers with ReLU activation
    out = layers.Dense(512, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(128, activation="relu")(out)

    # Output layer: produces 6D action
    # tanh ensures output is in [-1, 1], then scaled by upper_bound
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Scale output by action bounds (upper_bound = 1.0 for normalized actions)
    outputs = outputs * upper_bound

    model = tf.keras.Model(inputs, outputs)
    return model

# =====================================================================
#                     Critic Network
# =====================================================================
def get_critic():
    """
    Build the Critic (Value) network.

    Estimates state-action Q-value: (state, action) -> Q-value
    The critic learns an approximation of the optimal value function
    Q^π(s, a) ≈ E[r + γ*Q^π(s', π(s'))]

    Architecture:
    - State path: 6D input -> 64 -> 32 -> 32 units
    - Action path: 6D input -> 32 units
    - Merged: concatenate [state_features, action_features]
    - Dense layers: 256 -> 256 units
    - Output: scalar Q-value

    Why separate paths for state and action?
    - State features capture environment context
    - Action features help evaluate specific actions
    - Concatenation allows interaction between state and action information
    """
    # State pathway: processes state independently
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(64, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action pathway: processes action independently
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Concatenate state and action features
    # This allows the network to learn how different actions affect different states
    concat = layers.Concatenate()([state_out, action_out])

    # Fully connected layers on concatenated features
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)

    # Output: single Q-value for the state-action pair
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)
    return model

# =====================================================================
#                     Policy Execution
# =====================================================================
def policy(state, noise_object, add_noise=True):
    """
    Execute policy with optional exploration noise.

    Computes action from actor network and optionally adds OU noise
    for exploratory behavior during training.

    Args:
        state: Current state [x, y, z, goal_x, goal_y, goal_z] (must be batched)
        noise_object: OUActionNoise instance for exploration
        add_noise: Whether to add exploration noise (True for training, False for evaluation)

    Returns:
        Action clipped to valid range [lower_bound, upper_bound]
    """
    # Forward pass through actor network
    # squeeze removes batch dimension (1, action_size) -> (action_size,)
    sampled_actions = tf.squeeze(actor_model(state))

    # Sample OU noise for exploration
    noise = noise_object()

    # Add noise to actions during training (exploration)
    if add_noise:
        sampled_actions = sampled_actions.numpy() + noise

    # Ensure actions stay within valid bounds
    # Lower and upper bounds typically [-1, 1] for normalized actions
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

# =====================================================================
#                     Initialize Networks
# =====================================================================

# Create actor and critic networks
actor_model = get_actor()
critic_model = get_critic()

# Create target networks with same architecture
target_actor = get_actor()
target_critic = get_critic()

# Initialize target networks with actor/critic weights
# Targets will be updated slowly during training
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# =====================================================================
#                     Training Hyperparameters
# =====================================================================

# Learning rates for gradient descent
critic_lr = 1e-3  # Critic learns value function at this rate
actor_lr = 1e-4   # Actor learns policy at slower rate (for stability)

# Optimizers: Adam with specified learning rates
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 400

# Discount factor: how much future rewards matter
# gamma = 0.99 means future rewards discounted by 1% per step
# Closer to 1 = consider long-term rewards more
gamma = 0.99

# Soft update rate for target networks
# tau << 1 means target updates slowly (stable learning)
# Typical values: 1e-3 to 5e-3
tau = 5e-3

# Replay buffer: stores transitions for training
# Capacity 500k, batch size 128
buffer = Buffer(int(5e5), 128)

# =====================================================================
#                     Training/Evaluation Loop
# =====================================================================

# Storage for tracking learning progress
ep_reward_list = []         # Reward per episode
avg_reward_list = []        # Moving average of rewards
counter = 0                 # Count of successful episodes (reached goal)
avg_reward = 0

TRAIN = False  # Set to True to train, False to evaluate

if TRAIN:
    # Training exploration noise
    # std_dev = 0.2 gives moderate exploration
    std_dev = 0.2
    ou_noise = OUActionNoise(
        mean=np.zeros(num_actions),  # Zero mean noise
        std_deviation=float(std_dev) * np.ones(num_actions)
    )

    # Main training loop
    for ep in range(total_episodes):

        # Reset environment: random start position and random goal
        prev_state = env.reset()

        # Print episode info every 50 episodes
        if ep % 50 == 0:
            print('Episode Number', ep)
            # Print first 3 state components (x, y, z)
            print("Initial Position is", prev_state[0:3])
            print("===============================================================")
            # Print last 3 state components (goal_x, goal_y, goal_z)
            print("Target Position is", prev_state[3:6])
            print("===============================================================")
            print("Initial Kappas are ", [env.kappa1, env.kappa2, env.kappa3])
            print("===============================================================")
            print("Initial Phis are ", [env.phi1, env.phi2, env.phi3])
            print("===============================================================")

        episodic_reward = 0

        # Run episode for max 500 steps
        for i in range(500):
            # Add batch dimension for network input
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            # Get action from policy (with exploration noise)
            action = policy(tf_prev_state, ou_noise)

            # Execute action in environment
            # Reward function options:
            # - 'step_minus_euclidean_square': -distance^2 (smooth, large penalties for far states)
            # - 'step_minus_weighted_euclidean': -0.7*distance (weighted version)
            # - 'step_error_comparison': discrete rewards (-1, -0.5, 1)
            # - 'step_distance_based': shaped rewards based on distance bins
            state, reward, done, info = env.step(
                action[0],
                reward_function=config['reward']['function']
            )

            # Store experience in replay buffer
            buffer.record((prev_state, action, reward, state))

            # Accumulate episode reward
            episodic_reward += reward

            # Learn from random batch of past experiences
            buffer.learn()

            # Soft update target networks
            # Makes them slowly track the learning networks
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # Episode ends when goal is reached (done=True)
            if done:
                counter += 1
                break

            # Move to next state
            prev_state = state

        # Record episode rewards
        ep_reward_list.append(episodic_reward)

        # Compute moving average over last 100 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        if ep % 1 == 0:
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            time.sleep(0.5)
        avg_reward_list.append(avg_reward)

    print(f'{counter} times robot reached the target point in total {total_episodes} episodes')

    # Plot training curves
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.title("Moving Average Reward (100 episodes)")

    # Save average reward history
    with open('avg_reward_list.pickle', 'wb') as f:
        pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)

    # Plot episodic rewards
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(ep_reward_list)+1), ep_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Episodic Reward')
    plt.title("Per-Episode Rewards")
    plt.show()

    # Save episodic reward history
    with open('ep_reward_list.pickle', 'wb') as f:
        pickle.dump(ep_reward_list, f, pickle.HIGHEST_PROTOCOL)

    # Save trained model weights
    actor_model.save_weights("experiment/continuum_actor.h5")
    critic_model.save_weights("experiment/continuum_critic.h5")
    target_actor.save_weights("experiment/continuum_target_actor.h5")
    target_critic.save_weights("experiment/continuum_target_critic.h5")

    end_time = time.time() - start_time
    print('Total Workspace Overshoots: ', env.overshoot0)
    print('Total Goal Overshoots: ', env.overshoot1)
    print(f'Total Elapsed Time is {int(end_time)/60} minutes')

else:
    # Evaluation: load pre-trained weights and test policy
    actor_model.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_actor.h5")
    critic_model.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_critic.h5")
    target_actor.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_target_actor.h5")
    target_critic.load_weights(f"../Keras/{config['goal_type']}/{config['reward']['file']}/model/continuum_target_critic.h5")

    # Evaluation code would go here
    # Currently commented out - uncomment to visualize trained policy
