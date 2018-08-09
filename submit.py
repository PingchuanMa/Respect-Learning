import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import argparse

import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from tools.utils import load_state

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "[YOUR_CROWD_AI_TOKEN_HERE]"


def submit(identifier, policy_fn, seed, iter):

	client = Client(remote_base)

	# Create environment
	observation = client.env_create(crowdai_token)

	# IMPLEMENTATION OF YOUR CONTROLLER
	my_controller = train(identifier, policy_fn, 1, 500, seed, save_final=False)
	load_state(identifier, iter)

	while True:
	    [observation, reward, done, info] = client.env_step(my_controller(observation), True)
	    if done:
	        observation = client.env_reset()
	        if not observation:
	            break

	client.submit()


def main():

    logger.configure()
    parser = argparse.ArgumentParser(description='Submit.')
    parser.add_argument('--id', type=str, default='origin')
    parser.add_argument('--net', type=int, nargs='+', default=(128, 64))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iter', type=str, default='final')
    args = parser.parse_args()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_layer_sizes=args.net)

    submit(args.id, policy_fn, args.seed, args.iter)
