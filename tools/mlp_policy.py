from baselines.ppo1 import mlp_policy
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import tensorflow.contrib as tc
import gym
from baselines.common.distributions import make_pdtype


class MlpPolicy(mlp_policy.MlpPolicy):

    def _init(self, ob_space, ac_space, hid_layer_sizes, gaussian_fixed_var=True, noise_std=0.0, layer_norm=False, activation=tf.nn.relu):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i, size in enumerate(hid_layer_sizes):
                last_out = activation(tf.layers.dense(last_out, size, name="fc%i" % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i, size in enumerate(hid_layer_sizes):
                last_out = tf.layers.dense(last_out, size, name="fc%i" % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0))
                if layer_norm:
                    last_out = tc.layers.layer_norm(last_out, center=True, scale=True)
                noise = tf.random_normal(shape=tf.shape(last_out), mean=0.0, stddev=noise_std, dtype=tf.float32)
                last_out = activation(last_out + noise)
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final',
                                       kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])
