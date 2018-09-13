import numpy as np

from osim import env
import param
from util import state_desc_to_ob
from util import get_mirror_id
from gym import spaces

class ProstheticsEnv(env.ProstheticsEnv):

    def __init__(self, visualize = True, integrator_accuracy = 5e-5, bend_para = -0.9599310849999999, mirror=False ):
        
        self.mirror = mirror
        super().__init__(visualize, integrator_accuracy)
        self.bend_para = bend_para
        self.bend_base = np.exp( - np.square(self.bend_para) / 2 ) / ( 1 *  np.sqrt( 2 * np.pi )) 
        # self.mirror_id = None
        self.mirror = mirror
        if mirror:
            self.reset()
            # additional information are already added into the state descriptions in self.reset() where
            # self.get_observation are called 
            self.mirror_id = get_mirror_id(self.get_state_desc())

            self.action_space = ( [0.0] * (self.osim_model.get_action_space_size() + 3), [1.0] * (self.osim_model.get_action_space_size() + 3) )
            self.action_space = spaces.Box(np.array(self.action_space[0]), np.array(self.action_space[1]) )

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
            return 248
        return 167

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        rew_ori = 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        rew_speed = param.w_speed * rew_ori

        rew_straight = -param.w_straight * (state_desc["body_pos"]["pelvis"][2] ** 2)
        rew_straight -= param.w_straight * (state_desc["body_pos"]["head"][2] ** 2)
        rew_straight -= param.w_straight * (state_desc["body_pos"]["torso"][2] ** 2)

        # original left   -0.0835
        # original right   0.0835
        shift_base = 0.0835

        rew_straight -= 1.0 / 6 * param.w_straight * ((state_desc["body_pos"]["femur_r"][2] - shift_base ) ** 2)
        rew_straight -= 1.0 / 6 * param.w_straight * ((state_desc["body_pos"]["femur_l"][2] + shift_base ) ** 2)

        rew_straight -= 1.0 / 6 * param.w_straight * ((state_desc["body_pos"]["pros_tibia_r"][2] - shift_base ) ** 2)
        rew_straight -= 1.0 / 6 * param.w_straight * ((state_desc["body_pos"]["tibia_l"][2] + shift_base ) ** 2)

        rew_straight -= 1.0 / 6 * param.w_straight * ((state_desc["body_pos"]["pros_foot_r"][2] - shift_base ) ** 2)
        rew_straight -= 1.0 / 6 * param.w_straight * ((state_desc["body_pos"]["talus_l"][2] + shift_base ) ** 2)
        
        # rew_pose = param.w_pose * (9.0 + np.minimum(state_desc["body_pos"]["head"][0] - state_desc["body_pos"]["pelvis"][0], 0))
        # rew_fall = param.w_fall * (9.0 + np.minimum(state_desc["body_pos"]["head"][1] - state_desc["body_pos"]["pelvis"][1], 0))

        rew_bend_l = np.max( np.min( -state_desc["joint_pos"]["knee_l"][0] ,
                     state_desc["joint_pos"]["knee_l"][0] - 2 * self.bend_para ), 0 )
        rew_bend_r = np.max( np.min( -state_desc["joint_pos"]["knee_r"][0] ,
                     state_desc["joint_pos"]["knee_r"][0] - 2 * self.bend_para ), 0 )
        
        rew_bend = param.w_bend * ( rew_bend_l + rew_bend_r )

        rew_lean_back = -param.w_lean_back * np.clip( 
            ( state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["torso"][0] ) ,
            a_min = 0, a_max=None )

        rew_lean_back -= param.w_lean_back * np.clip( 
            ( state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0] ) ,
            a_min = 0, a_max=None )

        # rew_mirror = -param.w_mirror * (state_desc["body_pos"]["head"][2] ** 2)

        rew_total = rew_speed + rew_straight + rew_bend + rew_lean_back
        
        rew_total = param.rew_scale * (rew_total + param.rew_const)

        rew_all = {'original': rew_ori, 'speed': rew_speed, 'straight': rew_straight, 'bend': rew_bend, 'bend_r': rew_bend_r, 'bend_l': rew_bend_l}
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

