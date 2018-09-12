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


## Observation Space
### Extended Observation space
duplicate info of pros leg for left part of the model
duplicate info of right leg for right part of the model

## Action Space
0   abd_r
1   add_r
2   hamstrings_r
3   bifemsh_r
4   glut_max_r
5   iliopsoas_r
6   rect_fem_r
7   vasti_r
8   abd_l
9   add_l
10   hamstrings_l
11   bifemsh_l
12   glut_max_l
13   iliopsoas_l
14   rect_fem_l
15   vasti_l
16   gastroc_l
17   soleus_l
18   tib_ant_l 

### Extended Action Space
19   gastroc_r
20   soleus_r
21   tib_ant_r 
