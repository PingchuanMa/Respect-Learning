import numpy as np

from osim import env
import param
from util import state_desc_to_ob
from util import cascade_helper
from util import get_mirror_id
from gym import spaces
from reward import Reward

class ProstheticsEnv(env.ProstheticsEnv):

    def __init__(self, visualize=True, integrator_accuracy=5e-5, bend_para=-0.4,
                 mirror=False, reward_version=0, difficulty=0, fix_target=False,
                 no_acc=False, action_bias=0.0, target_adv=0, target_tau=0,
                 random_target=False, target_vx=1.25, clear_vz=False):
        
        self.mirror = mirror
        self.difficulty = difficulty
        self.fix_target = fix_target
        self.no_acc = no_acc
        self.action_bias = action_bias
        self.target_adv = target_adv
        self.target_tau = target_tau
        self.target_vel = None
        self.random_target = random_target
        self.target_vx = target_vx
        self.clear_vz = clear_vz
        super().__init__(visualize, integrator_accuracy, difficulty)
        self.bend_para = bend_para
        self.bend_base = np.exp( - np.square(self.bend_para) / 2 ) / ( 1 *  np.sqrt( 2 * np.pi ))
        if self.fix_target:
            self.time_limit = 512

        if mirror:
            self.reset()
            # additional information are already added into the state descriptions in self.reset() where
            # self.get_observation are called 
            self.mirror_id = get_mirror_id(self.get_state_desc(), self.difficulty, self.no_acc )

            self.action_space = ( [0.0] * (self.osim_model.get_action_space_size() + 3), [1.0] * (self.osim_model.get_action_space_size() + 3) )
            self.action_space = spaces.Box(np.array(self.action_space[0]), np.array(self.action_space[1]) )
        
        self.reward_set = Reward(self.bend_para, self.bend_base, difficulty)
        self.reward_func = getattr(self.reward_set, 'v' + str(reward_version))

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6 # or np.abs(state_desc["body_pos"]["pelvis"][2]) > 0.6   # encourage going straight
    
    def soft_update_target_vel( self, prev_target, current_target):
        
        if prev_target is None:
            return current_target
        return [ self.target_tau * p + (1-self.target_tau) * c for p,c in zip( prev_target, current_target ) ]

    def get_observation(self):
        state_desc = self.get_state_desc()
        self.target_vel = self.soft_update_target_vel( self.target_vel, state_desc["target_vel"])
        return state_desc_to_ob(state_desc, self.difficulty, self.mirror, self.no_acc, fix_target=self.fix_target,
            current_target_vel=self.target_vel, target_vx=self.target_vx, clear_vz=self.clear_vz)

    def get_cascade_arch(self):
        state_desc = self.get_state_desc()
        return cascade_helper(state_desc, self.difficulty, self.mirror)

    def get_observation_space_size(self):
        if self.prosthetic == True:
            if self.mirror:
                if self.no_acc:
                    return 404
                return 529
            if self.no_acc:
                return 326
            return 412
        return 167
        # state_desc = self.get_state_desc()
        # return len(state_desc_to_ob(state_desc, self.difficulty, self.mirror))

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        if self.difficulty == 0:
            rew_ori = self.reward_origin_round1( state_desc )
        else:
            rew_ori = self.reward_origin_round2( state_desc )

        self.reward_set.set_target_vel(state_desc)
        rew_all = self.reward_func(state_desc)
        rew_all['const'] = param.rew_const 
        rew_total = sum(rew_all.values()) - self.get_activation_penalty()
        rew_total *= param.rew_scale
        rew_all['original'] = rew_ori
        return rew_total, rew_all

    def reward_origin_round1(self, state_desc):
        return 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2

    def reward_origin_round2(self, state_desc):

        penalty = 0

        # Small penalty for too much activation (cost of transport)
        penalty += np.sum(np.array(self.osim_model.get_activations())**2) * 0.001

        # Big penalty for not matching the vector on the X,Z projection.
        # No penalty for the vertical axis
        penalty += (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0])**2
        penalty += (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2])**2
        
        # Reward for not falling
        reward = 10.0
        
        return reward - penalty 

    def get_activation_penalty(self):
        return np.sum(np.array(self.osim_model.get_activations())**2) * 0.001

    def step(self, action, project = True):
        self.prev_state_desc = self.get_state_desc()

        if self.mirror == True:
            action = self.action_process_mirror(action)
        
        action += self.action_bias

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

    def reset(self, project = True):
        if self.fix_target:
            target_x = 0.7 + np.random.random() if self.random_target else self.target_vx
            self.targets = np.array([[target_x, .0, .0] for _ in range(self.time_limit * 2)])
        else:
            self.generate_new_targets()
        self.osim_model.reset()
        if not project:
            return self.get_state_desc()
        else:
            return self.get_observation()

    def generate_new_targets_ori(self):
        super(ProstheticsEnv, self).generate_new_targets()

    def generate_new_targets(self):
        super(ProstheticsEnv, self).generate_new_targets()
        self.targets = np.concatenate([self.targets[self.target_adv:], np.repeat([self.targets[-1]], self.target_adv, axis=0)])


class TestProstheticsEnv(ProstheticsEnv):

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6

    def reset(self, project = True):
        self.generate_new_targets()
        self.osim_model.reset()
        if not project:
            return self.get_state_desc()
        else:
            return self.get_observation()

    def generate_new_targets(self):
        super(TestProstheticsEnv, self).generate_new_targets_ori()
