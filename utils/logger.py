"""
Logging utilities for tracking training metrics
"""
import os
import json
import csv
import numpy as np
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """
    Logger for tracking and saving training metrics
    """

    def __init__(self, log_dir, algorithm_name, seed):
        """
        Args:
            log_dir: Directory to save logs
            algorithm_name: Name of the algorithm
            seed: Random seed used
        """
        self.algorithm_name = algorithm_name
        self.seed = seed

        # Create algorithm-specific subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_subdir = Path(log_dir) / f"{algorithm_name}_seed{seed}_{timestamp}"
        self.log_subdir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics storage
        self.metrics = {
            'episode': [],
            'episode_reward': [],
            'episode_steps': [],
            'avg_reward': [],
            'std_reward': [],
            'eval_reward': []
        }

        # CSV file for continuous logging
        self.csv_path = self.log_subdir / 'training_metrics.csv'
        self._init_csv()

        print(f"✓ Logger initialized: {self.log_subdir}")

    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics.keys())

    def log_episode(self, episode, episode_reward, episode_steps,
                    avg_reward=None, std_reward=None, eval_reward=None):
        """
        Log metrics for a single episode

        Args:
            episode: Episode number
            episode_reward: Total reward for the episode
            episode_steps: Number of steps in the episode
            avg_reward: Average reward (rolling window)
            std_reward: Standard deviation of reward
            eval_reward: Evaluation reward (if available)
        """
        self.metrics['episode'].append(episode)
        self.metrics['episode_reward'].append(episode_reward)
        self.metrics['episode_steps'].append(episode_steps)
        self.metrics['avg_reward'].append(avg_reward if avg_reward is not None else episode_reward)
        self.metrics['std_reward'].append(std_reward if std_reward is not None else 0)
        self.metrics['eval_reward'].append(eval_reward if eval_reward is not None else 0)

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, episode_reward, episode_steps,
                avg_reward if avg_reward is not None else episode_reward,
                std_reward if std_reward is not None else 0,
                eval_reward if eval_reward is not None else 0
            ])

    def get_metrics(self):
        """Return all logged metrics"""
        return self.metrics

    def save_summary(self):
        """Save summary statistics to JSON"""
        if len(self.metrics['episode_reward']) == 0:
            return

        summary = {
            'algorithm': self.algorithm_name,
            'seed': self.seed,
            'total_episodes': len(self.metrics['episode']),
            'mean_reward': float(np.mean(self.metrics['episode_reward'])),
            'std_reward': float(np.std(self.metrics['episode_reward'])),
            'max_reward': float(np.max(self.metrics['episode_reward'])),
            'min_reward': float(np.min(self.metrics['episode_reward'])),
            'final_avg_reward': float(np.mean(self.metrics['episode_reward'][-100:])) if len(
                self.metrics['episode_reward']) >= 100 else None
        }

        summary_path = self.log_subdir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"✓ Summary saved: {summary_path}")
        return summary

    def print_progress(self, episode, episode_reward, avg_reward, eval_reward=None):
        """Print training progress"""
        msg = f"Episode {episode:4d} | Reward: {episode_reward:7.2f} | Avg: {avg_reward:7.2f}"
        if eval_reward is not None:
            msg += f" | Eval: {eval_reward:7.2f}"
        print(msg)


class ExperimentLogger:
    """
    Logger for managing multiple algorithm runs
    """

    def __init__(self, base_log_dir):
        """
        Args:
            base_log_dir: Base directory for all experiments
        """
        self.base_log_dir = Path(base_log_dir)
        self.algorithm_loggers = {}

    def create_logger(self, algorithm_name, seed):
        """
        Create a logger for a specific algorithm and seed

        Args:
            algorithm_name: Name of the algorithm
            seed: Random seed

        Returns:
            MetricsLogger instance
        """
        key = f"{algorithm_name}_seed{seed}"
        logger = MetricsLogger(self.base_log_dir, algorithm_name, seed)
        self.algorithm_loggers[key] = logger
        return logger

    def aggregate_results(self, algorithm_name):
        """
        Aggregate results across multiple seeds for an algorithm

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            Dictionary with aggregated statistics
        """
        # Find all loggers for this algorithm
        algo_loggers = [v for k, v in self.algorithm_loggers.items()
                        if k.startswith(algorithm_name)]

        if not algo_loggers:
            return None

        # Collect all episode rewards
        all_rewards = [logger.get_metrics()['episode_reward']
                       for logger in algo_loggers]

        # Calculate statistics
        min_length = min(len(rewards) for rewards in all_rewards)
        all_rewards_array = np.array([rewards[:min_length] for rewards in all_rewards])

        aggregated = {
            'mean_rewards': np.mean(all_rewards_array, axis=0),
            'std_rewards': np.std(all_rewards_array, axis=0),
            'num_seeds': len(algo_loggers),
            'episodes': list(range(1, min_length + 1))
        }

        return aggregated


if __name__ == "__main__":
    # Test logger
    print("Testing Logger...")

    logger = MetricsLogger("results/logs", "DQN", seed=42)

    # Simulate some episodes
    for ep in range(10):
        reward = np.random.randn() * 10 + 100
        steps = np.random.randint(50, 200)
        logger.log_episode(ep, reward, steps)

    # Save summary
    summary = logger.save_summary()
    print(f"\n✓ Summary statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")