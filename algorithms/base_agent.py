from abc import ABC, abstractmethod
import gymnasium as gym
import torch
import numpy as np


class BaseRLAgent(ABC):
    """
    Base class for all RL agents to ensure consistent interface
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, num_episodes, logger=None):
        """Train the agent"""
        pass

    @abstractmethod
    def predict(self, state, deterministic=True):
        """Predict action given state"""
        pass

    @abstractmethod
    def save(self, path):
        """Save model"""
        pass

    @abstractmethod
    def load(self, path):
        """Load model"""
        pass


class DQN_SB3_Wrapper(BaseRLAgent):
    """
    Wrapper for Stable-Baselines3 DQN

    Installation: pip install stable-baselines3
    Docs: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        from stable_baselines3 import DQN

        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.get('learning_rate', 1e-3),
            buffer_size=config.get('replay_buffer_size', 100000),
            learning_starts=config.get('learning_starts', 1000),
            batch_size=config.get('batch_size', 64),
            gamma=config.get('gamma', 0.99),
            target_update_interval=config.get('target_update_frequency', 10) * 1000,
            exploration_fraction=0.1,
            exploration_initial_eps=config.get('epsilon_start', 1.0),
            exploration_final_eps=config.get('epsilon_end', 0.01),
            policy_kwargs=dict(net_arch=config.get('hidden_dims', [128, 128])),
            verbose=1
        )

    def train(self, num_episodes, logger=None):
        """
        Train using SB3's learn method
        Note: SB3 uses timesteps, not episodes
        """
        # Estimate timesteps from episodes
        max_steps = self.config.get('max_steps_per_episode', 500)
        total_timesteps = num_episodes * max_steps

        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=10,
            progress_bar=True
        )

    def predict(self, state, deterministic=True):
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        from stable_baselines3 import DQN
        self.model = DQN.load(path, env=self.env)


class DDPG_SB3_Wrapper(BaseRLAgent):
    """
    Wrapper for Stable-Baselines3 TD3 (improved DDPG)

    Installation: pip install stable-baselines3
    Docs: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html

    Note: Using TD3 instead of DDPG as it's more stable and performs better
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        from stable_baselines3 import TD3

        self.model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.get('actor_lr', 1e-4),
            buffer_size=config.get('replay_buffer_size', 100000),
            learning_starts=config.get('learning_starts', 1000),
            batch_size=config.get('batch_size', 64),
            gamma=config.get('gamma', 0.99),
            tau=config.get('tau', 0.005),
            policy_kwargs=dict(
                net_arch=dict(
                    pi=config.get('actor_hidden_dims', [256, 256]),
                    qf=config.get('critic_hidden_dims', [256, 256])
                )
            ),
            verbose=1
        )

    def train(self, num_episodes, logger=None):
        max_steps = self.config.get('max_steps_per_episode', 500)
        total_timesteps = num_episodes * max_steps

        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=10,
            progress_bar=True
        )

    def predict(self, state, deterministic=True):
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        from stable_baselines3 import TD3
        self.model = TD3.load(path, env=self.env)


class WDDPG_Wrapper(BaseRLAgent):
    """
    Wrapper for Weighted DDPG

    Options:
    1. Find implementation on GitHub: https://github.com/search?q=weighted+ddpg
    2. Modify DDPG with Prioritized Experience Replay
    3. Implement based on paper

    Example GitHub sources:
    - Search "weighted ddpg pytorch" on GitHub
    - Look for implementations of prioritized experience replay + DDPG
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        # Import your W-DDPG implementation
        # Option A: From external source
        # Option B: Modified DDPG with PER from utils/replay_buffer.py
        pass

    def train(self, num_episodes, logger=None):
        pass

    def predict(self, state, deterministic=True):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

def create_agent(algorithm_name, env, config):
    """
    Factory function to create agents

    Args:
        algorithm_name: 'dqn', 'ddpg', or 'w_ddpg'
        env: Gymnasium environment
        config: Configuration dictionary

    Returns:
        Agent instance
    """
    agents = {
        'dqn': DQN_SB3_Wrapper,
        'ddpg': DDPG_SB3_Wrapper,
        'w_ddpg': WDDPG_Wrapper,
    }

    if algorithm_name not in agents:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    return agents[algorithm_name](env, config)


if __name__ == "__main__":
    """
    Example usage of the wrappers
    """
    # Create environment
    env = gym.make("CartPole-v1")

    # Configuration
    config = {
        'learning_rate': 1e-3,
        'batch_size': 64,
        'replay_buffer_size': 100000,
        'gamma': 0.99,
        'learning_starts': 1000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'hidden_dims': [128, 128],
        'max_steps_per_episode': 500
    }

    # Create agent using factory
    agent = create_agent('dqn', env, config)

    print(f"✓ Agent created: {type(agent).__name__}")
    print(f"✓ Model type: {type(agent.model).__name__}")


    agent.train(num_episodes=100)

    agent.save("models/dqn_model")

    agent.load("models/dqn_model")

    action = agent.predict(env.reset()[0])

    env.close()