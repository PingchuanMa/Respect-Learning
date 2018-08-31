import numpy as np

from osim import env
import param
from util import state_desc_to_ob

class ProstheticsEnv(env.ProstheticsEnv):

    def __init__(self, visualize = True, integrator_accuracy = 5e-5):
        super().__init__(visualize, integrator_accuracy)

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6 or np.abs(state_desc["body_pos"]["pelvis"][2]) > 0.6   # encourage going straight

    def get_observation(self):
        state_desc = self.get_state_desc()
        return state_desc_to_ob(state_desc)

    def get_observation_space_size(self):
        if self.prosthetic == True:
            return 331
        return 167

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        rew_ori = 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        rew_speed = param.w_speed * rew_ori
        rew_straight = -param.w_straight * (state_desc["body_pos"]["pelvis"][2] ** 2)
        # rew_pose = param.w_pose * (9.0 + np.minimum(state_desc["body_pos"]["head"][0] - state_desc["body_pos"]["pelvis"][0], 0))
        # rew_fall = param.w_fall * (9.0 + np.minimum(state_desc["body_pos"]["head"][1] - state_desc["body_pos"]["pelvis"][1], 0))
        rew_total = rew_speed + rew_straight
        rew_total = param.rew_scale * (rew_total + param.rew_const)
        return rew_total, rew_ori

    def step(self, action, project = True):
        self.prev_state_desc = self.get_state_desc()        
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()
        
        rew_total, rew_ori = self.reward()
        return [ obs, rew_total, self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit), {'rew_ori': rew_ori} ]


class TestProstheticsEnv(ProstheticsEnv):

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6

