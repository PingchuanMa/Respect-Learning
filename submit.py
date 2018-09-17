import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import argparse
import multiprocessing
import tensorflow as tf
import numpy as np

import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from tools.utils import load_state
from tools import mlp_policy
from run import train
from util import state_desc_to_ob
import param

# Settings
remote_base = "http://grader.crowdai.org:1729"
with open('./token.txt', 'r') as f:
    crowdai_token = f.readline()


def submit(identifier, policy_fn, seed, iter, mirror):

    client = Client(remote_base)

    # Create environment
    observation = client.env_create(crowdai_token, env_id="ProstheticsEnv")

    # IMPLEMENTATION OF YOUR CONTROLLER
    pi = train(identifier, policy_fn, 1, 1, seed, mirror=mirror, save_final=False, play=True, bend=0, ent=0, symcoeff=0)
    load_state(identifier, iter)

    while True:
        ob = state_desc_to_ob(observation, mirror=mirror)
        action = pi.act(False, np.array(ob))[0].tolist()
        if mirror:
            action = action[:-3]
        # for _ in range(param.action_repeat):
        [observation, reward, done, info] = client.env_step(action, True)
            # if done:
                # break
        if done:
            observation = client.env_reset()
            if not observation:
                break

    client.submit()


def main():

    parser = argparse.ArgumentParser(description='Submit.')
    parser.add_argument('--id', type=str, default='origin')
    parser.add_argument('--net', type=int, nargs='+', default=(256, 128, 64))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iter', type=str, default='final')
    parser.add_argument('--mirror', default=False, action='store_true')
    args = parser.parse_args()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_layer_sizes=args.net)

    #tf configs
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    submit(args.id, policy_fn, args.seed, args.iter, args.mirror)


if __name__ == '__main__':
    main()
