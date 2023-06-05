# CityFlowRL

CityFlowRL is a reinforcement learning traffic signal control project that utilizes machine learning algorithms to optimize traffic flow and minimize congestion. The project involves training an intelligent traffic signal controller using deep reinforcement learning algorithms.


## Project Structure

```
├── README.md - This file.
├── CityFlowEnv # Custom gym environment directory
└── RL
    ├── frontend # For manual replaying.
    │
    ├── tests # For testing diffrent reinforcement learning algorithms.
    │   ├── replay.py # Script for automatic replaying after testing.
    │   ├── testA2C.py
    │   ├── testDQN.py
    │   ├── testPPO.py
    │   └── testRandom.py
    └── configs # Datasets for testing on various real-life and synthetic scenarios.
```

## Usage

Navigate to CityFlowEnv directory and run setup

```bash
  pip install .
```

Then run Navigate to RL/tests and run testRandom.py to check that everything is working.