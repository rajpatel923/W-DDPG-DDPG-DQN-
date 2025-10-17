"""
Configuration loader utility for RL experiments
"""
import yaml
import os
from pathlib import Path


def load_config(config_path="config/hyperparameters.yaml"):
    """
    Load hyperparameters from YAML configuration file

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_directories(config):
    """
    Create necessary directories for logging and saving models

    Args:
        config: Configuration dictionary
    """
    log_dir = config['logging']['log_dir']
    model_dir = config['logging']['model_dir']
    plot_dir = config['logging']['plot_dir']

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    print(f"✓ Created directories:")
    print(f"  - Logs: {log_dir}")
    print(f"  - Models: {model_dir}")
    print(f"  - Plots: {plot_dir}")


def get_algorithm_config(config, algorithm_name):
    """
    Get combined configuration for a specific algorithm

    Args:
        config: Full configuration dictionary
        algorithm_name: Name of the algorithm ('dqn', 'ddpg', or 'w_ddpg')

    Returns:
        dict: Combined configuration with common and algorithm-specific params
    """
    if algorithm_name not in ['dqn', 'ddpg', 'w_ddpg']:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Combine common parameters with algorithm-specific ones
    algo_config = {
        **config['common'],
        **config[algorithm_name],
        'environment': config['environment'],
        'training': config['training'],
        'logging': config['logging']
    }

    return algo_config


if __name__ == "__main__":
    # Test the configuration loader
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"\nAvailable algorithms: {list(config.keys())}")

    # Test directory creation
    create_directories(config)

    # Test algorithm-specific config
    dqn_config = get_algorithm_config(config, 'dqn')
    print(f"\nDQN Config keys: {list(dqn_config.keys())}")