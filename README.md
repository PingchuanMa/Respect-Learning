# Respect-Learning

## Setup
After following setup procedure on official website, run setup.sh for installing required packages.

## Usage
Run run.py for local training and testing, run submit.py for online submission.

## To-dos
1. Speed-up physical simulator (current 20+steps/s)
2. Customize environment
3. Benchmark RL algorithm (DDPG, PPO, TRPO, ...)
4. Action repeat / enlarge timestep (4x)
5. Benchmark activation function (SELU, RELU, ELU, ...)
6. Reward scaling (works?)
7. Action noise & parameter noise
8. Layer normalization
9. Symmetry handling
10. Benchmark 1D conv / FC
11. Reward shaping (encourage bending knees, encourage leaning forward, penalize falling, use velocity instead of distance ...)
12. Observation extension