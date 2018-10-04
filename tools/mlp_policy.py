from baselines.ppo1 import mlp_policy
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import tensorflow.contrib as tc
import gym
from baselines.common.distributions import make_pdtype


class MlpPolicy(mlp_policy.MlpPolicy):

    def _init(self, ob_space, ac_space, hid_layer_sizes, gaussian_fixed_var=True, noise_std=0.0, layer_norm=False, activation=tf.nn.relu, **kwargs):
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


class MpcPolicy(mlp_policy.MlpPolicy):

    def _init(self, ob_space, ac_space, end_points, gaussian_fixed_var=True,
              noise_std=0.0, layer_norm=False, activation=tf.nn.relu, **kwargs):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        def mpc_conv(layer, num_filter, kernel_size):
            layer = tf.layers.conv2d(layer, num_filter, kernel_size,
                                     use_bias=not layer_norm,
                                     kernel_initializer=tc.layers.xavier_initializer())
            if layer_norm:
                layer = tc.layers.layer_norm(layer, center=True, scale=True)
            return activation(layer)

        def mpc_fc(layer, units=None):
            units = layer.get_shape().as_list()[1] if units is None else units
            layer = tf.layers.dense(layer, units, use_bias=not layer_norm,
                                    kernel_initializer=tc.layers.xavier_initializer())
            if layer_norm:
                layer = tc.layers.layer_norm(layer, center=True, scale=True)
            return activation(layer)

        def build_net(layer):
            len_body = end_points["body"][1] - end_points["body"][0]
            body_tensor = layer[:, end_points["body"][0]:end_points["body"][1]]
            joint_pos_tensor = layer[:, end_points["joint_pos"][0]:end_points["joint_pos"][1]]
            joint_vel_tensor = layer[:, end_points["joint_vel"][0]:end_points["joint_vel"][1]]
            joint_acc_tensor = layer[:, end_points["joint_acc"][0]:end_points["joint_acc"][1]]
            muscle_tensor = layer[:, end_points["muscle"][0]:end_points["muscle"][1]]
            force_tensor = layer[:, end_points["force"][0]:end_points["force"][1]]
            misc_tensor = layer[:, end_points["misc"][0]:end_points["misc"][1]]

            body_tensor = tf.reshape(body_tensor, (-1, len_body // (3 * 6), 6, 3))

            body_tensor_1 = mpc_conv(body_tensor, 32, (len_body // (3 * 6), 1))
            body_tensor_1 = mpc_conv(body_tensor_1, 32, (1, 1)) + body_tensor_1
            body_tensor_1 = mpc_conv(body_tensor_1, 3, (1, 1))
            body_tensor_1 = tf.layers.flatten(body_tensor_1)

            body_tensor_2 = mpc_conv(body_tensor, 32, (1, 6))
            body_tensor_2 = mpc_conv(body_tensor_2, 32, (1, 1)) + body_tensor_2
            body_tensor_2 = mpc_conv(body_tensor_2, 3, (1, 1))
            body_tensor_2 = tf.layers.flatten(body_tensor_2)

            joint_pos_tensor = mpc_fc(joint_pos_tensor)
            joint_pos_tensor = mpc_fc(joint_pos_tensor) + joint_pos_tensor
            joint_vel_tensor = mpc_fc(joint_vel_tensor)
            joint_vel_tensor = mpc_fc(joint_vel_tensor) + joint_vel_tensor
            joint_acc_tensor = mpc_fc(joint_acc_tensor)
            joint_acc_tensor = mpc_fc(joint_acc_tensor) + joint_acc_tensor

            muscle_tensor = mpc_fc(muscle_tensor)
            muscle_tensor = mpc_fc(muscle_tensor) + muscle_tensor
            force_tensor = mpc_fc(force_tensor)
            force_tensor = mpc_fc(force_tensor) + force_tensor
            misc_tensor = mpc_fc(misc_tensor)
            misc_tensor = mpc_fc(misc_tensor) + misc_tensor

            emsumble = tf.concat([body_tensor_1, body_tensor_2, joint_pos_tensor, joint_vel_tensor, joint_acc_tensor,
                                  muscle_tensor, force_tensor, misc_tensor], axis=1)
            emsumble = mpc_fc(emsumble, 256)
            emsumble = mpc_fc(emsumble) + emsumble
            emsumble = mpc_fc(emsumble) + emsumble
            return emsumble

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = build_net(obz)
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=tc.layers.xavier_initializer())[:, 0]

        with tf.variable_scope('pol'):
            last_out = build_net(obz)
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
