# Respect-Learning

## Setup
After following setup procedure on official website, run setup.sh for installing required packages.

## Usage
Run run.py for local training and testing, run submit.py for online submission.

## To-dos
1. Speed-up physical simulator (current 20+steps/s)
2. Customize environment
3. Benchmark RL algorithm (DDPG, PPO, TRPO, ...)
4. Benchmark activation function (SELU, RELU, ELU, ...)
5. Reward scaling (works?)
6. Action noise & parameter noise
7. Layer normalization
8. Symmetry handling
9. Benchmark 1D conv / FC
10. Reward shaping (encourage bending knees)
11. Observation transformation (absolute position to relative position for body parts except pelvis)