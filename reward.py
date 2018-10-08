import numpy as np
import param

class Reward():

    @staticmethod
    def calc_distance_square( target_vel, input_vec ):
        vel_vec = target_vel[0::2]
        input_vec = input_vec[0::2]

        vel_vec = np.array( vel_vec )
        input_vec = np.array( input_vec )

        distance = np.cross( input_vec, vel_vec ) / np.linalg.norm ( vel_vec ) 

        return np.square(distance)


    def __init__(self, bend_para, bend_base):
        self.bend_para = bend_para
        self.bend_base = bend_base

    def v0(self, state_desc, difficulty ):
        rew_all = {}
        return rew_all

    def v1(self, state_desc, difficulty ):
        
        if difficulty > 0:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = [3,0,0]

        rew_straight = -param.w_straight * Reward.calc_distance_square( target_vel, state_desc["body_pos"]["pelvis"])
        rew_all = {'straight': rew_straight}
        return rew_all

    def v2(self, state_desc, difficulty ):

        if difficulty > 0:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = [3,0,0]

        rew_straight = -param.w_straight * Reward.calc_distance_square( target_vel, state_desc["body_pos"]["pelvis"])
        rew_bend_l = np.exp(-np.square(state_desc["joint_pos"]["knee_l"][0] - self.bend_para) / 2) / (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend_r = np.exp(-np.square(state_desc["joint_pos"]["knee_r"][0] - self.bend_para) / 2) / (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend = param.w_bend * (rew_bend_l + rew_bend_r )
        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all

    def v3(self, state_desc, difficulty ):

        if difficulty > 0:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = [3,0,0]

        shift_base = 0.0835
        rew_straight = -param.w_straight * (
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["pelvis"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["head"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["torso"]) +
            ((state_desc["body_pos"]["femur_r"][2] - shift_base) ** 2) +
            ((state_desc["body_pos"]["femur_l"][2] + shift_base) ** 2) +
            ((state_desc["body_pos"]["pros_tibia_r"][2] - shift_base) ** 2) +
            ((state_desc["body_pos"]["tibia_l"][2] + shift_base) ** 2) +
            ((state_desc["body_pos"]["pros_foot_r"][2] - shift_base) ** 2) +
            ((state_desc["body_pos"]["talus_l"][2] + shift_base) ** 2))
        rew_bend = -param.w_bend * (
            np.clip(state_desc["joint_pos"]["knee_l"][0], self.bend_para, 0) + 
            np.clip(state_desc["joint_pos"]["knee_r"][0], self.bend_para, 0))
        rew_lean_back = -param.w_lean_back * (
            np.maximum((state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["torso"][0]), 0) +
            np.maximum((state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0]), 0))
        rew_all = {'straight': rew_straight, 'bend': rew_bend, 'lean_back': rew_lean_back}
        return rew_all

    def v4(self, state_desc, difficulty ):

        if difficulty > 0:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = [3,0,0]

        rew_straight = -param.w_straight * (
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["pelvis"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["head"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["torso"]))

        rew_bend = param.w_bend * (
                max( min( -state_desc["joint_pos"]["knee_l"][0] ,
                     state_desc["joint_pos"]["knee_l"][0] - 2 * self.bend_para ), 0. ) + 
                max( min( -state_desc["joint_pos"]["knee_r"][0] ,
                     state_desc["joint_pos"]["knee_r"][0] - 2 * self.bend_para ), 0. ))

        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all
    
    def v5(self, state_desc, difficulty ):

        if difficulty > 0:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = [3,0,0]

        rew_straight = -param.w_straight * (
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["pelvis"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["head"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["torso"]))

        rew_bend = -param.w_bend * (
            np.clip(state_desc["joint_pos"]["knee_l"][0], self.bend_para, 0) + 
            np.clip(state_desc["joint_pos"]["knee_r"][0], self.bend_para, 0))

        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all

    def v6(self, state_desc, difficulty ):

        if difficulty > 0:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = [3,0,0]


        rew_straight = -param.w_straight * (
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["pelvis"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["head"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["torso"]))
        rew_bend_l = np.exp(-np.square(state_desc["joint_pos"]["knee_l"][0] - self.bend_para) / 2) / (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend_r = np.exp(-np.square(state_desc["joint_pos"]["knee_r"][0] - self.bend_para) / 2) / (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend = param.w_bend * (rew_bend_l + rew_bend_r )
        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all

    def v7(self, state_desc, difficulty ):

        if difficulty > 0:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = [3,0,0]

        rew_straight = -param.w_straight * (
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["pelvis"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["head"]) +
            Reward.calc_distance_square( target_vel, state_desc["body_pos"]["torso"]))

        rew_speed_fix = state_desc["target_vel"][0] ** 2 + state_desc["target_vel"][2] ** 2

        rew_speed_fix -= (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) ** 2 + \
                         (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) ** 2
        
        rew_speed_fix *= param.w_speed

        rew_bend = param.w_bend * (
                max( min( -state_desc["joint_pos"]["knee_l"][0] ,
                     state_desc["joint_pos"]["knee_l"][0] - 2 * self.bend_para ), 0. ) + 
                max( min( -state_desc["joint_pos"]["knee_r"][0] ,
                     state_desc["joint_pos"]["knee_r"][0] - 2 * self.bend_para ), 0. ))

        rew_all = {'straight': rew_straight, 'bend': rew_bend, 'speed': rew_speed_fix}
        return rew_all