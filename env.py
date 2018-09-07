import numpy as np

from osim import env
import param
from util import state_desc_to_ob

class ProstheticsEnv(env.ProstheticsEnv):

    def __init__(self, visualize = True, integrator_accuracy = 5e-5, bend_para = -0.9599310849999999):
        super().__init__(visualize, integrator_accuracy)
        self.bend_para = bend_para
        self.bend_base = np.exp( - np.square(self.bend_para) / 2 ) / ( 1 *  np.sqrt( 2 * np.pi )) 

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6 or np.abs(state_desc["body_pos"]["pelvis"][2]) > 0.6   # encourage going straight

    def get_observation(self):
        state_desc = self.get_state_desc()
        return state_desc_to_ob(state_desc)

    def get_observation_space_size(self):
        if self.prosthetic == True:
            return 334
        return 167

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        rew_ori = 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        rew_speed = param.w_speed * rew_ori
        rew_straight = -param.w_straight * (state_desc["body_pos"]["pelvis"][2] ** 2)
        # rew_bend_l = np.exp( - np.square(state_desc["joint_pos"]["knee_l"][0] - self.bend_para) / 2 ) / ( 1 *  np.sqrt( 2 * np.pi )) - self.bend_base
        # rew_bend_r = np.exp( - np.square(state_desc["joint_pos"]["knee_r"][0] - self.bend_para) / 2 ) / ( 1 *  np.sqrt( 2 * np.pi )) - self.bend_base
        # rew_bend = param.w_bend * (rew_bend_l + rew_bend_r )

        rew_total = rew_speed + rew_straight
        rew_total = param.rew_scale * (rew_total + param.rew_const)

        rew_all = {'original': rew_ori, 'speed': rew_speed, 'straight': rew_straight}
        return rew_total, rew_all

    def step(self, action, project = True):
        self.prev_state_desc = self.get_state_desc()        
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()
        
        rew_total, rew_all = self.reward()
        return [ obs, rew_total, self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit), rew_all]


class TestProstheticsEnv(ProstheticsEnv):

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6

