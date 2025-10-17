"""
Replay buffer for experience replay in RL algorithms
"""
import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling transitions
    """

    def __init__(self, capacity, device='cpu'):
        """
        Args:
            capacity: Maximum size of the buffer
            device: Device to store tensors ('cpu' or 'cuda')
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (optional for advanced implementations)
    """

    def __init__(self, capacity, alpha=0.6, device='cpu'):
        """
        Args:
            capacity: Maximum size of the buffer
            alpha: Prioritization exponent
            device: Device to store tensors
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.device = device

    def push(self, state, action, reward, next_state, done, priority=None):
        """Add transition with priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority if priority is not None else max_priority)

    def sample(self, batch_size, beta=0.4):
        """Sample batch with importance sampling weights"""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # Test replay buffer
    print("Testing Replay Buffer...")

    buffer = ReplayBuffer(capacity=1000)

    # Add some dummy transitions
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = False
        buffer.push(state, action, reward, next_state, done)

    print(f"✓ Buffer size: {len(buffer)}")

    # Test sampling
    if len(buffer) >= 32:
        batch = buffer.sample(32)
        print(f"✓ Sampled batch shapes:")
        print(f"  States: {batch[0].shape}")
        print(f"  Actions: {batch[1].shape}")
        print(f"  Rewards: {batch[2].shape}")
        print(f"  Next States: {batch[3].shape}")
        print(f"  Dones: {batch[4].shape}")