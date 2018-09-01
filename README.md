# Respect-Learning

## Setup
After following setup procedure on official website, run setup.sh for installing required packages.

## Usage
Run run.py for local training and testing, run submit.py for online submission.

## To-dos
### Simulator
1. Understand observation, especially physical properties of joint
2. Figure out the reason why action repeat slows things down so much
### Reward Shaping
1. Encourage bending knees
2. Activate muscle more for speed up
3. Encourage more aggressive stepping, currently the agent is too cautious to make a big step
### RL Tricks
1. Discretize action value
2. Action noise & parameter noise
3. Layer normalization
4. Observation engineering
### Benchmarks
1. Network structure
2. Activation function (SELU, RELU, ELU, ...)
3. 1D conv / FC
