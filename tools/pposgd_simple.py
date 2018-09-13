from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque, OrderedDict
from tools.utils import save_state, save_rewards, load_state


def traj_segment_generator(pi, env, horizon, stochastic, mirror_id=None, action_repeat=1):
    mirror = mirror_id is not None
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_ret_all = {}
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_rets_all = {}
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    if mirror:
        mirror_obs = obs.copy()
        mirror_acs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, np.array(ob))
        if mirror:
            mirror_ob = ob[mirror_id[0]]
            if len(mirror_id)>2:
                mirror_ob *= mirror_id[2]
            mirror_ac, _ = pi.act(stochastic, np.array(mirror_ob))
            mirror_ac = mirror_ac[mirror_id[1]]
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            seg_dict = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                        "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_rets_all" : ep_rets_all, "ep_lens" : ep_lens}
            if mirror:
                seg_dict['mirror_ob'] = mirror_obs
                seg_dict['mirror_ac'] = mirror_acs
            yield seg_dict
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_rets_all = {}
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        if mirror:
            mirror_obs[i] = mirror_ob
            mirror_acs[i] = mirror_ac

        rew = 0
        rew_all = {}
        for ai in range(action_repeat):
            ob, r, new, r_all = env.step(ac)
            rew = rew * ai / (ai + 1) + r / (ai + 1)
            if ai == 0:
                rew_all = r_all
            elif r_all:
                for name, val in r_all.items():
                    rew_all[name] = rew_all[name] * ai / (ai + 1) + val / (ai + 1)
            if new:
                break
        rews[i] = rew

        cur_ep_ret += rew
        if not cur_ep_ret_all:
            cur_ep_ret_all = rew_all
        else:
            for name, val in rew_all.items():
                cur_ep_ret_all[name] += val

        cur_ep_len += 1
        if not ep_rets_all and cur_ep_ret_all:
            for name in cur_ep_ret_all.keys():
                ep_rets_all[name] = []
        if new:
            ep_rets.append(cur_ep_ret)
            if ep_rets_all:
                for name, val in cur_ep_ret_all.items():
                    ep_rets_all[name].append(val)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            if cur_ep_ret_all:
                for name, _ in cur_ep_ret_all.items():
                    cur_ep_ret_all[name] = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          identifier,
          save_result=True,
          save_interval=100,
          reward_list=[],
          reward_ori_list=[],
          cont=False,
          iter, play, action_repeat=1):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    mirror = hasattr(env, 'mirror_id')
    mirror_id = env.mirror_id if mirror else None
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    if mirror:
        mirror_ob = U.get_placeholder(name="mirror_ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        mirror_ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    sym_loss = 4 * tf.reduce_mean(tf.square(ac - mirror_ac)) if mirror else 0
    total_loss = pol_surr + pol_entpen + vf_loss + sym_loss

    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]
    if mirror:
        losses.append(sym_loss)
        loss_names.append("sym_loss")

    var_list = pi.get_trainable_variables()
    inputs = [ob, ac, atarg, ret, lrmult]
    if mirror:
        inputs += [mirror_ob, mirror_ac]
    lossandgrad = U.function(inputs, losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function(inputs, losses)

    if play:
        return pi

    if cont:
        load_state(identifier, iter)
    else:
        U.initialize()
        iter = 0
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True, mirror_id=mirror_id, action_repeat=action_repeat)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = int(iter)
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_all = {}

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        if mirror:
            mirror_ob, mirror_ac = seg["mirror_ob"], seg["mirror_ac"]

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d_dict = dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret)
        if mirror:
            d_dict["mirror_ob"] = mirror_ob
            d_dict["mirror_ac"] = mirror_ac
        d = Dataset(d_dict, shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                batches = [batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult]
                if mirror:
                    batches += [batch["mirror_ob"], batch["mirror_ac"]]
                *newlosses, g = lossandgrad(*batches)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)

        losses = []
        for batch in d.iterate_once(optim_batchsize):
            batches = [batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult]
            if mirror:
                batches += [batch["mirror_ob"], batch["mirror_ac"]]
            newlosses = compute_losses(*batches)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)

        # for (lossval, name) in zipsame(meanlosses, loss_names):
        #     logger.record_tabular("loss_"+name, lossval)
        # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = [seg["ep_lens"], seg["ep_rets"]] # local values
        names = []
        if seg["ep_rets_all"]:
            for name, val_list in OrderedDict(sorted(seg["ep_rets_all"].items())).items():
                names.append(name)
                lrlocal.append(val_list)
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lrs = list(map(flatten_lists, zip(*listoflrpairs)))
        lens, rews = lrs[0], lrs[1]
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        for name, rews_i in zip(names, lrs[2:]):
            if name not in rewbuffer_all:
                rewbuffer_all[name] = deque(maxlen=100)
            rewbuffer_all[name].extend(rews_i)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        if rewbuffer_all:
            for name, val in rewbuffer_all.items():
                logger.record_tabular("EpRewMean(" + name + ")", np.mean(rewbuffer_all[name]))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

            reward_list.append(np.mean(rewbuffer))
            if rewbuffer_all:
                reward_ori_list.append(np.mean(rewbuffer_all['original']))
            if save_result and iters_so_far % save_interval == 0:
                save_state(identifier, iters_so_far)
                save_rewards(reward_list, identifier, iters_so_far)
                if rewbuffer_all:
                    save_rewards(reward_ori_list, identifier + '_ori', iters_so_far)
                logger.log('Model and reward saved')

    return pi


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
