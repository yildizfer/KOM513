# 3D DDPG Implementation Summary

## Overview
Updated DDPG (Deep Deterministic Policy Gradient) implementations for controlling a 3-segment soft continuum robot in 3D space. Both Keras/TensorFlow and PyTorch versions are provided with detailed comments explaining all key components.

## Key Changes from 2D to 3D

### State Space
- **2D**: 4D observation [x, y, goal_x, goal_y]
- **3D**: 6D observation [x, y, z, goal_x, goal_y, goal_z]
- Automatically calculated from environment: `num_states = env.observation_space.shape[0] * 2`

### Action Space
- **2D**: 3D action [κ̇₁, κ̇₂, κ̇₃] (3 curvature rates)
- **3D**: 6D action [κ̇₁, κ̇₂, κ̇₃, φ̇₁, φ̇₂, φ̇₃] (3 curvature rates + 3 bending angle rates)
- Automatically calculated: `num_actions = env.action_space.shape[0]`

### Jacobian Matrix
- **2D**: 2×3 Jacobian (2D velocity, 3 DOF)
- **3D**: 6×6 Jacobian (6D velocity including orientation, 6 DOF)

### Reward Function
- **2D**: 2D Euclidean distance `error = sqrt((goal_x-x)² + (goal_y-y)²)`
- **3D**: 3D Euclidean distance `error = sqrt((goal_x-x)² + (goal_y-y)² + (goal_z-z)²)`

---

## Files

### Keras Implementation (TensorFlow)

#### `DDPG_3D.py` (NEW - Fully Commented)
Main training script with extensive documentation:

**Key Classes:**
1. **OUActionNoise**: Ornstein-Uhlenbeck exploration noise
   - Temporally correlated noise for smoother exploration
   - Formula: `x_t = x_{t-1} + θ(μ - x_{t-1})Δt + σ√Δt N(0,1)`
   - Hyperparameters: θ=0.15 (mean reversion), σ=0.2 (noise scale)

2. **Buffer**: Experience replay buffer
   - Stores transitions: (state, action, reward, next_state)
   - Circular indexing with fixed capacity (500k)
   - Batch size: 128
   - Breaks temporal correlations in training data

3. **Network Functions:**
   - `get_actor()`: 6D input → 512 → 256 → 128 → 6D output (tanh)
   - `get_critic()`: State path (64→32→32) + Action path (32) → 256 → 256 → 1

**Key Hyperparameters:**
- Actor learning rate: 1e-4
- Critic learning rate: 1e-3  
- Discount factor (γ): 0.99
- Soft update rate (τ): 5e-3
- Training episodes: 400
- Max steps per episode: 500

**Training Process:**
```python
for ep in range(total_episodes):
    state = env.reset()  # Random position and goal
    for step in range(500):
        action = policy(state) + noise  # Exploration
        next_state, reward, done = env.step(action)
        buffer.record((state, action, reward, next_state))
        buffer.learn()  # Update networks from random batch
        update_target(...)  # Soft update target networks
        state = next_state
        if done: break
```

#### `DDPG.py` (Original - Modified for 3D)
The original file automatically works with 3D because:
- State/action sizes calculated dynamically from environment
- All network layers use these variables
- Only hardcoded indices (for printing) need updating

**What Changed in DDPG.py:**
- Lines 315, 317: Updated print statements to handle 6D state
  - Old: `prev_state[0:2]` → New: `prev_state[0:3]`
  - Old: `prev_state[2:4]` → New: `prev_state[3:6]`

---

### PyTorch Implementation

#### `ddpg_3D.py` (NEW - Fully Commented)
Training script with extensive documentation:
- Cleaner structure than Keras version
- More modular for custom modifications
- Better debugging capabilities

**Key Changes:**
- Line 54: `Agent(state_size=6, action_size=6, ...)`
- Lines 103, 109: Updated state slicing for printing

#### `ddpg.py` (Original - Needs Update)
**Changes Required:**
```python
# Line 34: CHANGE THIS
agent = Agent(state_size=4, action_size=3, random_seed=10)
# TO:
agent = Agent(state_size=6, action_size=6, random_seed=10)

# Lines 50, 52: Update print statements
# OLD: state[0:2], state[2:4]
# NEW: state[0:3], state[3:6]
```

#### `ddpg_agent.py` (No Changes Needed!)
The Agent class uses state_size and action_size as parameters:
```python
class Agent():
    def __init__(self, state_size, action_size, random_seed):
        self.actor_local = Actor(state_size, action_size, random_seed)
        self.critic_local = Critic(state_size, action_size, random_seed)
        ...
```
So it automatically works with 6D!

#### `model_3D.py` (NEW - Fully Commented)
Neural network architectures:

**Actor:**
```
Input (6) 
  ↓ [Linear, ReLU]
  64 units
  ↓ [Linear, ReLU]
  32 units
  ↓ [Linear, ReLU]
  16 units
  ↓ [Linear, ReLU]
  8 units
  ↓ [Linear, Tanh]
Output (6) ∈ [-1, 1]
```

**Critic:**
```
State (6)                    Action (6)
  ↓                            ↓
[Linear, LeakyReLU]          [Linear, LeakyReLU]
  ↓                            ↓
64 units                     32 units
  ↓                            ↓
32 units                       (processed separately)
  ├─────────────────────────┬──┘
  Concatenate [32 + 6 = 38]
  ↓
[Linear, LeakyReLU]
32 units
  ↓
[Linear, LeakyReLU]
16 units
  ↓
[Linear]
Output (1) = Q-value
```

#### `model.py` (Original - Works Automatically)
No changes needed! Uses state_size and action_size parameters.

---

## Network Architecture Explanation

### Why These Sizes?

**Actor Network (Policy):**
- Start at 512 (large for feature extraction from 6D state)
- Progressively shrink: 512 → 256 → 128
- Output: 6D actions with tanh ∈ [-1, 1]
- Smaller networks can be unstable; larger captures richer policies

**Critic Network (Q-Function):**
- State path: 64 → 32 → 32 (moderate feature extraction)
- Action path: 32 units (simple action encoding)
- Concatenate: [state_features, action_features]
- Final layers: 256 → 256 → 1
- Larger final layers because Q-value estimation is harder than policy

### Activation Functions

| Layer | Activation | Why |
|-------|-----------|-----|
| Hidden Actor | ReLU | Fast training, no saturation |
| Output Actor | Tanh | Constrains actions to [-1, 1] |
| Hidden Critic (state) | LeakyReLU | Prevents dead neurons (small negative slope) |
| Hidden Critic (merged) | LeakyReLU | Better gradient flow for value estimation |
| Output Critic | None | Q-values can be any real number |

### Weight Initialization

- **Hidden layers**: `Uniform(-1/√fan_in, 1/√fan_in)`
  - Scales with input dimension to prevent vanishing/exploding gradients
  
- **Output layers**: `Uniform(-3e-3, 3e-3)`
  - Very small for stable initial policy and Q-value estimates
  - Critical for DDPG stability!

---

## DDPG Algorithm Flow

### Per Episode:
1. **Reset environment**: Get random start and goal positions
2. **Initialize noise**: Reset OU noise process
3. **For each step (up to 500):**
   - **Policy**: a = π(s) + noise (exploration)
   - **Environment**: s', r, done = env.step(a)
   - **Store**: Add (s, a, r, s') to replay buffer
   - **Learn**: Sample batch from buffer, update networks
   - **Soft Update**: θ_target = τθ_local + (1-τ)θ_target
   - If done: break episode

### Network Updates (Per Batch):

**Critic Update:**
```
Loss = MSE(Q(s,a), r + γ·Q_target(s', π_target(s')))
∇L_critic = gradient of loss w.r.t. critic weights
θ_critic ← θ_critic - α∇L_critic
```
Critic learns to predict future returns accurately.

**Actor Update:**
```
Loss = -E[Q(s, π(s))]  (negative because we maximize)
∇L_actor = gradient of loss w.r.t. actor weights
θ_actor ← θ_actor - α∇L_actor
```
Actor learns to produce actions that maximize Q-values.

**Soft Update:**
```
θ_target = 0.005·θ_local + 0.995·θ_target
```
Target networks change slowly to stabilize learning.

---

## Hyperparameter Tuning Guide

### If Training is Unstable:
- **Reduce actor learning rate**: 1e-4 → 1e-5 (slower learning = more stable)
- **Increase τ (soft update rate)**: 5e-3 → 1e-2 (target networks adapt faster)
- **Reduce noise std_dev**: 0.2 → 0.1 (less exploration randomness)

### If Convergence is Slow:
- **Increase actor learning rate**: 1e-4 → 1e-3 (faster learning)
- **Increase batch size**: 64 → 128 (more data per update)
- **Increase buffer capacity**: Sample larger diversity

### If Agent Gets Stuck:
- **Increase noise std_dev**: 0.2 → 0.3 (more exploration)
- **Decrease discount factor**: 0.99 → 0.95 (focus on immediate rewards)
- **Increase critic learning rate**: 1e-3 → 2e-3

---

## Usage

### Keras Version:
```python
# Training
TRAIN = True
python DDPG_3D.py

# Evaluation
TRAIN = False
python DDPG_3D.py
```

### PyTorch Version:
```python
# Training
TRAIN = True
python ddpg_3D.py

# Evaluation
TRAIN = False
python ddpg_3D.py
```

---

## Key Insights for 3D Control

1. **6D Action Space Complexity**: With 6 dimensions instead of 3, the agent must learn to coordinate curvature AND bending angle for each segment. This requires more training data.

2. **Larger Jacobian**: The 6×6 Jacobian provides more flexibility but also more control complexity. The algorithm must learn the coupled dynamics.

3. **Reward Signal**: 3D distance reward provides continuous feedback, helping the agent learn smoother trajectories.

4. **Exploration**: OU noise is crucial - the agent needs systematic exploration to discover 3D reaching strategies.

---

## Comparison: Keras vs PyTorch

| Aspect | Keras | PyTorch |
|--------|-------|---------|
| Code length | ~400 lines | ~250 lines |
| Readability | Functional style | Imperative OOP |
| Flexibility | Good | Excellent |
| Debugging | Built-in graph mode | Eager execution by default |
| Performance | Similar | Similar |
| Custom layers | Good | Excellent |

**Recommendation**: Use PyTorch for research/experimentation, Keras for production deployment.

---

## Common Issues & Fixes

**Issue**: NaN in loss
- **Cause**: Network weights exploding or learning rate too high
- **Fix**: Reduce learning rates, check weight initialization

**Issue**: Reward not improving
- **Cause**: Exploration noise too high or network not learning
- **Fix**: Reduce noise, increase learning rate, check if environment works

**Issue**: Training diverges
- **Cause**: Unstable network updates
- **Fix**: Increase τ (soft update), reduce learning rate

**Issue**: Very slow convergence
- **Cause**: Network architecture too small or learning rates too low
- **Fix**: Increase hidden layer sizes, increase learning rates

---

## References

1. Lillicrap et al. "Continuous control with deep reinforcement learning" (DDPG paper)
2. Hannan & Walker "Kinematics and the implementation of an elephant's trunk manipulator"
3. TensorFlow/PyTorch documentation
