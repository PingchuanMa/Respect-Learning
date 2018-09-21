import numpy as np
import param

class Reward():

    def __init__(self, bend_para, bend_base):
        self.bend_para = bend_para
        self.bend_base = bend_base

    def v0(self, state_desc):
        rew_all = {}
        return rew_all

    def v1(self, state_desc):
        rew_straight = -param.w_straight * (state_desc["body_pos"]["pelvis"][2] ** 2)
        rew_all = {'straight': rew_straight}
        return rew_all

    def v2(self, state_desc):
        rew_straight = -param.w_straight * (state_desc["body_pos"]["pelvis"][2] ** 2)
        rew_bend_l = np.exp(-np.square(state_desc["joint_pos"]["knee_l"][0] - self.bend_para) / 2) / (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend_r = np.exp(-np.square(state_desc["joint_pos"]["knee_r"][0] - self.bend_para) / 2) / (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend = param.w_bend * (rew_bend_l + rew_bend_r )
        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all

    def v3(self, state_desc):
        shift_base = 0.0835
        rew_straight = -param.w_straight * (
            (state_desc["body_pos"]["pelvis"][2] ** 2) +
            (state_desc["body_pos"]["head"][2] ** 2) +
            (state_desc["body_pos"]["torso"][2] ** 2) +
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
