import sys

import tensorflow as tf
import os
import json
from mpi4py import MPI
import datetime
import numpy as np

import os
result_path = os.path.dirname(os.path.abspath(__file__)) + '/../results/'


def get_session():
    return tf.get_default_session()


def load_state(identifier, iters_so_far):
    fname = result_path + 'model/' + identifier + '_' + str(iters_so_far) + '.ckpt'
    saver = tf.train.Saver()
    saver.restore(get_session(), fname)


def save_state(identifier, iters_so_far):
    fname = result_path + 'model/' + identifier + '_' + str(iters_so_far) + '.ckpt'
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(get_session(), fname)


def save_params(params, env_id):
    save_path = result_path + "param/"
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + env_id + ".json", "w") as f:
        json.dump(params, f)


def load_params(env_id):
    with open(result_path + "param/" + env_id + ".json", "r") as f:
        params = dict(json.load(f))
    return params


def save_rewards(reward_list, identifier, iters_so_far):
    reward_list = np.array(reward_list).tolist()
    save_path = result_path + "reward/"
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + identifier + '_' + str(iters_so_far) + ".json", "w") as f:
        json.dump(reward_list, f)


def load_rewards(identifier, iters_so_far):
    load_path = result_path + "reward/"
    with open(load_path + identifier + '_' + str(iters_so_far) + '.json', 'r') as f:
        reward_list = json.load(f)
    return reward_list


def tensorboard_summary(name, value, ite):
    pass