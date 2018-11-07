import sys

import tensorflow as tf
import os
import json
from mpi4py import MPI
import datetime
import numpy as np
#from tensorboardX import SummaryWriter

import os
result_path = os.path.dirname(os.path.abspath(__file__)) + '/../results/'


def get_session():
    return tf.get_default_session()


def load_state(identifier, iters_so_far):
    fname = os.path.join(result_path, 'model', identifier, '%s.ckpt' % iters_so_far)
    saver = tf.train.Saver()
    saver.restore(get_session(), fname)


def load_scoped_state(identifier, iters_so_far):
    fname = os.path.join(result_path, 'model', identifier, '%s.ckpt' % iters_so_far)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=identifier))
    saver.restore(get_session(), fname)


def save_state(identifier, iters_so_far):
    fname = os.path.join(result_path, 'model', identifier, '%s.ckpt' % iters_so_far)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(get_session(), fname)


def save_params(params, env_id):
    save_path = os.path.join(result_path, 'param')
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, '%s.json' % env_id), "w") as f:
        json.dump(params, f)


def load_params(env_id):
    with open(os.path.join(result_path, 'param', '%s.json' % env_id), "r") as f:
        params = dict(json.load(f))
    return params


def save_rewards(reward_list, identifier, iters_so_far):
    reward_list = np.array(reward_list).tolist()
    save_path = os.path.join(result_path, 'reward', identifier)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, '%s.json' % iters_so_far), "w") as f:
        json.dump(reward_list, f)


def load_rewards(identifier, iters_so_far):
    file_path = os.path.join(result_path, 'reward', identifier, '%s.json' % iters_so_far)
    if not os.path.isfile(file_path):
        return []
    with open(file_path, 'r') as f:
        reward_list = json.load(f)
    return reward_list


def get_tb_writer(identifier):
    #if MPI.COMM_WORLD.Get_rank() == 0:
    #    os.makedirs(os.path.join(result_path, 'tensorboard', identifier), exist_ok=True)
    #    return SummaryWriter(log_dir=os.path.join(result_path, 'tensorboard', identifier))
    #else:
    return None

def tb_summary(writer, name, value, ite, domain=None):
    #if MPI.COMM_WORLD.Get_rank() == 0:
    #    writer.add_scalar(name if domain is None else '%s/%s' % (domain, name), value, ite)
    pass

def close_tb_writer(writer):
    #if MPI.COMM_WORLD.Get_rank() == 0:
        #writer.close()
    pass
