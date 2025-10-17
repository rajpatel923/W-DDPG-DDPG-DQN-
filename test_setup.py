"""
Test script to verify the RL comparison project setup
Run this after setting up your environment to ensure everything works
"""

import sys
import torch
import gymnasium as gym
import numpy as np
from pathlib import Path


def check_imports():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("CHECKING PACKAGE IMPORTS")
    print("=" * 60)

    packages = {
        'torch': torch,
        'gymnasium': gym,
        'numpy': np,
    }

    try:
        import yaml
        packages['pyyaml'] = yaml
    except ImportError:
        print("❌ pyyaml not installed")
        return False

    try:
        import matplotlib
        packages['matplotlib'] = matplotlib
    except ImportError:
        print("❌ matplotlib not installed")
        return False

    all_good = True
    for name, package in packages.items():
        try:
            version = getattr(package, '__version__', 'unknown')
            print(f"✓ {name:15s} version: {version}")
        except Exception as e:
            print(f"❌ {name:15s} error: {e}")
            all_good = False

    return all_good


def check_directories():
    """Check if required directories exist"""
    print("\n" + "=" * 60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 60)

    required_dirs = [
        'config',
        'algorithms',
        'utils',
        'results',
        'results/logs',
        'results/models',
        'results/plots'
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            all_exist = False

    return all_exist


def check_config():
    """Check if configuration file exists and is valid"""
    print("\n" + "=" * 60)
    print("CHECKING CONFIGURATION")
    print("=" * 60)

    config_path = Path("config/hyperparameters.yaml")

    if not config_path.exists():
        print(f"❌ Configuration file missing: {config_path}")
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required_keys = ['environment', 'training', 'common', 'dqn', 'ddpg', 'w_ddpg', 'logging']
        for key in required_keys:
            if key in config:
                print(f"✓ Config section: {key}")
            else:
                print(f"❌ Missing config section: {key}")
                return False

        return True

    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def test_environments():
    """Test if environments can be created"""
    print("\n" + "=" * 60)
    print("TESTING GYMNASIUM ENVIRONMENTS")
    print("=" * 60)

    environments = {
        'CartPole-v1': 'discrete',
        'Pendulum-v1': 'continuous'
    }

    all_good = True
    for env_name, action_type in environments.items():
        try:
            env = gym.make(env_name)
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.close()

            print(f"✓ {env_name:20s} ({action_type})")
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")

        except Exception as e:
            print(f"❌ {env_name:20s} - Error: {e}")
            all_good = False

    return all_good


def test_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("CHECKING GPU/CUDA")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("ℹ CUDA not available - will use CPU (slower but works fine)")

    return True


def test_utils():
    """Test if utility modules can be imported"""
    print("\n" + "=" * 60)
    print("TESTING UTILITY MODULES")
    print("=" * 60)

    utils_modules = [
        ('utils.config_loader', 'Configuration loader'),
        ('utils.networks', 'Neural networks'),
        ('utils.replay_buffer', 'Replay buffer'),
        ('utils.logger', 'Logger')
    ]

    all_good = True
    for module_name, description in utils_modules:
        try:
            __import__(module_name)
            print(f"✓ {description:25s} ({module_name})")
        except ImportError as e:
            print(f"❌ {description:25s} - Import error: {e}")
            all_good = False

    return all_good


def main():
    """Run all tests"""
    print("\n")
    print("*" * 60)
    print("RL ALGORITHMS COMPARISON - SETUP VERIFICATION")
    print("*" * 60)
    print()

    results = {
        'Imports': check_imports(),
        'Directories': check_directories(),
        'Configuration': check_config(),
        'Environments': test_environments(),
        'GPU/CUDA': test_cuda(),
        'Utilities': test_utils()
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED! You're ready to start implementing algorithms.")
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues above.")
    print("=" * 60)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())