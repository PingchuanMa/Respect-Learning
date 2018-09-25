import numpy as np

from osim import env
import param
from util import state_desc_to_ob
from util import get_mirror_id
from gym import spaces
from reward import Reward

class ProstheticsEnv(env.ProstheticsEnv):

    def __init__(self, visualize = True, integrator_accuracy = 5e-5, bend_para=-0.4, mirror=False, reward_version=0 ):
        
        self.mirror = mirror
        super().__init__(visualize, integrator_accuracy)
        self.bend_para = bend_para
        self.bend_base = np.exp( - np.square(self.bend_para) / 2 ) / ( 1 *  np.sqrt( 2 * np.pi )) 

        if mirror:
            self.reset()
            # additional information are already added into the state descriptions in self.reset() where
            # self.get_observation are called 
            self.mirror_id = get_mirror_id(self.get_state_desc())

            self.action_space = ( [0.0] * (self.osim_model.get_action_space_size() + 3), [1.0] * (self.osim_model.get_action_space_size() + 3) )
            self.action_space = spaces.Box(np.array(self.action_space[0]), np.array(self.action_space[1]) )
        
        self.reward_set = Reward(self.bend_para, self.bend_base)
        self.reward_func = getattr(self.reward_set, 'v' + str(reward_version))

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6 or np.abs(state_desc["body_pos"]["pelvis"][2]) > 0.6   # encourage going straight

    def get_observation(self):
        state_desc = self.get_state_desc()
        return state_desc_to_ob(state_desc, self.mirror)

    def get_observation_space_size(self):
        if self.prosthetic == True:
            if self.mirror:
                return 407
            return 326
        return 167

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        rew_ori = 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        rew_speed = param.w_speed * rew_ori
        rew_all = self.reward_func(state_desc)
        rew_all['speed'] = rew_speed
        rew_total = sum(rew_all.values())
        rew_total = param.rew_scale * (rew_total + param.rew_const)
        rew_all['original'] = rew_ori
        return rew_total, rew_all

    def step(self, action, project = True):
        self.prev_state_desc = self.get_state_desc()        

        if self.mirror == True:
            action = self.action_process_mirror(action)

        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()
        
        rew_total, rew_all = self.reward()
        return [ obs, rew_total, self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit), rew_all]

    def action_process_mirror(self, action):
        return action[:-3]


class TestProstheticsEnv(ProstheticsEnv):

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6

