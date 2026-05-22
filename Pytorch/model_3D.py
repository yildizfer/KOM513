"""
PyTorch Neural Network Architectures for DDPG

Actor: Maps states -> deterministic actions
Critic: Maps (state, action) -> Q-value (state-action value)

3D Version:
- Actor input: 6D state, output: 6D action
- Critic input: 6D state + 6D action, output: scalar Q-value
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    Initialize layer weights uniformly from small interval based on fan-in.

    This initialization helps prevent vanishing/exploding gradients.
    The interval size is proportional to 1/sqrt(fan_in) for better scaling.

    Args:
        layer: PyTorch nn.Linear layer

    Returns:
        (lower_bound, upper_bound) tuple for uniform initialization
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# =====================================================================
#                     Actor Network
# =====================================================================

class Actor(nn.Module):
    """
    Actor (Policy) Network for DDPG.

    Maps states to continuous actions via deterministic policy.
    Uses batch normalization between hidden layers for improved convergence.

    Architecture for 3D:
    Input (6) -> FC1 (64) -> FC2 (32) -> FC3 (16) -> FC4 (8) -> Output (6, tanh)

    Why these layer sizes?
    - Start moderately large (64) to capture state features
    - Progressively shrink towards action dimension
    - Small output layer helps stability with tanh saturation
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32, fc3_units=16, fc4_units=8):
        """
        Initialize actor network.

        Args:
            state_size: Dimension of state space (6 for 3D)
            action_size: Dimension of action space (6 for 3D)
            seed: Random seed for reproducibility
            fc*_units: Number of units in each fully connected layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, action_size)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize network weights.

        Uses careful initialization to balance learning speed and stability:
        - Hidden layers: uniform distribution based on fan-in
        - Output layer: very small uniform [-3e-3, 3e-3] for stable initial policy
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        # Small initialization for output layer (critical for DDPG stability)
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forward pass: state -> action

        Args:
            state: State tensor, shape (batch_size, state_size)

        Returns:
            action: Deterministic action tensor, shape (batch_size, action_size)
                    Values in [-1, 1] via tanh activation
        """
        # ReLU hidden layers for feature extraction
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # tanh output: constrains actions to [-1, 1]
        # This matches normalized action space
        return torch.tanh(self.fc5(x))


# =====================================================================
#                     Critic Network
# =====================================================================

class Critic(nn.Module):
    """
    Critic (Value) Network for DDPG.

    Estimates Q-value function: Q(s, a) = expected return for taking action a in state s

    Key architectural choice: State and action are processed separately before concatenation.
    This allows the network to learn state features independently from action-value coupling.

    Architecture for 3D:
    State (6) -> FC1 (64) -> 32 units
    Action (6) -> FC1 (32 units)
    Concatenate -> FC2 (32) -> FC3 (16) -> Output (1)
    """

    def __init__(self, state_size, action_size, seed, fcs1_units=64, fc2_units=32, fc3_units=16):
        """
        Initialize critic network.

        Args:
            state_size: Dimension of state space (6 for 3D)
            action_size: Dimension of action space (6 for 3D)
            seed: Random seed
            fcs1_units: Units in first state processing layer
            fc2_units: Units after concatenation
            fc3_units: Units in final hidden layer before output
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # State processing stream
        self.fcs1 = nn.Linear(state_size, fcs1_units)

        # After concatenating state and action features (fcs1_units + action_size)
        # This layer integrates state and action information
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)

        # Further value function estimation
        self.fc3 = nn.Linear(fc2_units, fc3_units)

        # Output: scalar Q-value
        self.fc4 = nn.Linear(fc3_units, 1)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize network weights for stable learning."""
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # Output layer: small weights for stable value estimates
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forward pass: (state, action) -> Q-value

        Args:
            state: State tensor, shape (batch_size, state_size)
            action: Action tensor, shape (batch_size, action_size)

        Returns:
            Q-value: Scalar value tensor, shape (batch_size, 1)
                     Represents estimated discounted future reward
        """
        # Process state independently first
        # leaky_relu allows small negative gradients (prevents dying ReLU)
        xs = F.leaky_relu(self.fcs1(state))

        # Concatenate state features with action
        # This interaction term is key: Q-value depends on both state AND action
        x = torch.cat((xs, action), dim=1)

        # Process combined state-action features
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        # Output Q-value: no activation on output layer
        # Q-values can be positive or negative depending on reward
        return self.fc4(x)
