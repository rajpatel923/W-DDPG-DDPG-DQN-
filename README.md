# RL Algorithms Comparison

Compare W-DDPG, DDPG, and DQN on continuous control tasks.

## Setup
```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

## Run
```bash
# Train all algorithms
python train.py

# Train specific algorithm
python train.py --algo ddpg

# Custom config
python train.py --config my_config.yaml

# Plot results
python plot_results.py
```

## Results

Results saved in `results/` directory with plots comparing all three algorithms.