"""
Neural network architectures for RL algorithms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Q-Network (DQN) or Value Network
    """

    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    """
    Actor network for DDPG and W-DDPG (continuous action space)
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], max_action=1.0):
        super(Actor, self).__init__()

        layers = []
        prev_dim = state_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer with tanh activation
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.network(state)


class Critic(nn.Module):
    """
    Critic network for DDPG and W-DDPG
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(Critic, self).__init__()

        # Q1 architecture
        layers_q1 = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers_q1.append(nn.Linear(prev_dim, hidden_dim))
            layers_q1.append(nn.ReLU())
            prev_dim = hidden_dim

        layers_q1.append(nn.Linear(prev_dim, 1))
        self.q1_network = nn.Sequential(*layers_q1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_network(sa)
        return q1


def initialize_weights(module):
    """
    Initialize network weights using Xavier uniform initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    # Test networks
    print("Testing network architectures...")

    # Test MLP (for DQN)
    mlp = MLP(input_dim=4, output_dim=2, hidden_dims=[128, 128])
    mlp.apply(initialize_weights)
    print(f"✓ MLP created: {sum(p.numel() for p in mlp.parameters())} parameters")

    # Test Actor
    actor = Actor(state_dim=3, action_dim=1, hidden_dims=[256, 256])
    actor.apply(initialize_weights)
    print(f"✓ Actor created: {sum(p.numel() for p in actor.parameters())} parameters")

    # Test Critic
    critic = Critic(state_dim=3, action_dim=1, hidden_dims=[256, 256])
    critic.apply(initialize_weights)
    print(f"✓ Critic created: {sum(p.numel() for p in critic.parameters())} parameters")

    # Test forward pass
    test_state = torch.randn(1, 4)
    test_output = mlp(test_state)
    print(f"✓ MLP forward pass successful: output shape {test_output.shape}")