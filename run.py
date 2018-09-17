import multiprocessing
import tensorflow as tf
from baselines import logger
from baselines.common import set_global_seeds
import argparse
from mpi4py import MPI
import numpy as np

import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from env import ProstheticsEnv, TestProstheticsEnv

from tools import pposgd_simple
from tools import mlp_policy
from tools.utils import *
from tools.plot_rewards import plot_rewards
from util import state_desc_to_ob_idx
import param


def train(identifier, policy_fn, num_timesteps, steps_per_iter, seed, bend, ent, symcoeff, mirror, cont=False, iter=None, save_final=True, play=False):

    env = ProstheticsEnv(visualize=False, integrator_accuracy=param.accuracy, bend_para=bend, mirror=mirror)

    if cont:
        assert iter is not None
        reward_list = load_rewards(identifier, iter)
    else:
        reward_list = []

    set_global_seeds(seed + MPI.COMM_WORLD.Get_rank())
    timesteps_per_actorbatch = np.ceil(steps_per_iter / MPI.COMM_WORLD.Get_size()).astype(int)
    
    #learn
    pi = pposgd_simple.learn(env, policy_fn,
                             max_timesteps=num_timesteps,
                             timesteps_per_actorbatch=timesteps_per_actorbatch,
                             clip_param=0.2, entcoeff=ent,
                             symcoeff= symcoeff,
                             optim_epochs=10,
                             optim_stepsize=3e-4,
                             optim_batchsize=64,
                             gamma=0.99,
                             lam=0.95,
                             schedule='linear',
                             identifier=identifier,
                             save_result=True,
                             save_interval=50,
                             reward_list=reward_list,
                             cont=cont,
                             play=play,
                             iter=iter,
                             action_repeat=param.action_repeat)
    env.close()

    if save_final:
        #save states in the end
        curr_time = datetime.datetime.now().strftime('%y%m%d%H%M%S')
        save_rewards(reward_list, identifier, 'final')
        plot_rewards(reward_list, identifier)
        save_state(identifier, 'final')

    return pi


def test(identifier, policy_fn, seed, iter, mirror):
    
    pi = train(identifier, policy_fn, 1, 1, seed, bend=0, ent=0, symcoeff=0, mirror=mirror, save_final=False, play=True)
    load_state(identifier, iter)
    env = TestProstheticsEnv(visualize=True, mirror=mirror)

    observation = env.reset()
    reward = 0
    reward_ori = 0
    while True:
        action = pi.act(False, np.array(observation))[0]
        rew = 0
        rew_ori = 0
        for ai in range(param.action_repeat):
            observation, r, done, info = env.step(action)
            # print(r)
            r_ori = info['original']
            rew = rew * ai / (ai + 1) + r / (ai + 1)
            rew_ori = rew_ori * ai / (ai + 1) + r_ori / (ai + 1)
            if done:
                break
        reward += rew
        reward_ori += rew_ori
        if done:
            print('reward', reward)
            print('reward_ori', reward_ori)
            break
            # observation = env.reset()
            # reward = 0
            # reward_ori = 0


def main():

    logger.configure()
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('--id', type=str, default='origin')
    parser.add_argument('--step', type=int, default=1e9)
    parser.add_argument('--bend', type=float, default=-0.9599310849999999)
    parser.add_argument('--ent', type=float, default=0.001)
    parser.add_argument('--sym', type=float, default=0.001)
    parser.add_argument('--step_per_iter', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cont', default=False, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--iter', type=str, default='final')
    parser.add_argument('--net', type=int, nargs='+', default=(256, 128, 64))
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

    #train/test
    if not args.play:
        train(identifier=args.id, policy_fn=policy_fn, num_timesteps=args.step, steps_per_iter=args.step_per_iter, seed=args.seed, cont=args.cont, iter=args.iter, bend=args.bend, ent=args.ent, symcoeff=args.sym, mirror=args.mirror)
    else:
        test(identifier=args.id, policy_fn=policy_fn, seed=args.seed, iter=args.iter, mirror=args.mirror)


if __name__ == '__main__':
    main()
