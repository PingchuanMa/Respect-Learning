import numpy as np
import param

class Reward():

    @staticmethod
    def calc_distance_square(target_vel, input_vec):
        vel_vec = np.array(target_vel[0::2])
        input_vec = np.array(input_vec[0::2])
        distance = np.cross(input_vec, vel_vec) / np.linalg.norm(vel_vec) 
        return np.square(distance)

    def __init__(self, bend_para, bend_base, difficulty):
        self.bend_para = bend_para
        self.bend_base = bend_base
        self.difficulty = difficulty
        self.target_vel = None

    def set_target_vel(self, state_desc):
        self.target_vel = np.array(state_desc["target_vel"]) if self.difficulty > 0 else np.array([3, 0, 0])

    def _rew_straight_v0(self, state_desc):
        return -param.w_straight * Reward.calc_distance_square(self.target_vel, state_desc["body_pos"]["pelvis"])

    def _rew_straight_v1(self, state_desc):
        return -param.w_straight * (
            Reward.calc_distance_square(self.target_vel, state_desc["body_pos"]["pelvis"]) +
            Reward.calc_distance_square(self.target_vel, state_desc["body_pos"]["head"]) +
            Reward.calc_distance_square(self.target_vel, state_desc["body_pos"]["torso"]))

    def _rew_straight_v2(self, state_desc, shift_base):
        return -param.w_straight * (
            Reward.calc_distance_square(self.target_vel, state_desc["body_pos"]["pelvis"]) +
            Reward.calc_distance_square(self.target_vel, state_desc["body_pos"]["head"]) +
            Reward.calc_distance_square(self.target_vel, state_desc["body_pos"]["torso"]) +
            ((state_desc["body_pos"]["femur_r"][2] - shift_base) ** 2) +
            ((state_desc["body_pos"]["femur_l"][2] + shift_base) ** 2) +
            ((state_desc["body_pos"]["pros_tibia_r"][2] - shift_base) ** 2) +
            ((state_desc["body_pos"]["tibia_l"][2] + shift_base) ** 2) +
            ((state_desc["body_pos"]["pros_foot_r"][2] - shift_base) ** 2) +
            ((state_desc["body_pos"]["talus_l"][2] + shift_base) ** 2))

    def v0(self, state_desc):

        rew_all = {}
        return rew_all

    def v1(self, state_desc):

        rew_straight = self._rew_straight_v0(state_desc)

        rew_all = {'straight': rew_straight}
        return rew_all

    def v2(self, state_desc):

        rew_straight = self._rew_straight_v0(state_desc)

        rew_bend_l = np.exp(-np.square(state_desc["joint_pos"]["knee_l"][0] - self.bend_para) / 2) / \
            (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend_r = np.exp(-np.square(state_desc["joint_pos"]["knee_r"][0] - self.bend_para) / 2) / \
            (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend = param.w_bend * (rew_bend_l + rew_bend_r )

        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all

    def v3(self, state_desc):

        rew_straight = self._rew_straight_v2(state_desc, shift_base=0.0835)

        rew_bend = -param.w_bend * (
            np.clip(state_desc["joint_pos"]["knee_l"][0], self.bend_para, 0) + 
            np.clip(state_desc["joint_pos"]["knee_r"][0], self.bend_para, 0))

        rew_lean_back = -param.w_lean_back * (
            np.maximum((state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["torso"][0]), 0) +
            np.maximum((state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0]), 0))

        rew_all = {'straight': rew_straight, 'bend': rew_bend, 'lean_back': rew_lean_back}
        return rew_all

    def v4(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_bend = param.w_bend * (
                max( min( -state_desc["joint_pos"]["knee_l"][0] ,
                     state_desc["joint_pos"]["knee_l"][0] - 2 * self.bend_para ), 0. ) + 
                max( min( -state_desc["joint_pos"]["knee_r"][0] ,
                     state_desc["joint_pos"]["knee_r"][0] - 2 * self.bend_para ), 0. ))

        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all
    
    def v5(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_bend = -param.w_bend * (
            np.clip(state_desc["joint_pos"]["knee_l"][0], self.bend_para, 0) + 
            np.clip(state_desc["joint_pos"]["knee_r"][0], self.bend_para, 0))

        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all

    def v6(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_bend_l = np.exp(-np.square(state_desc["joint_pos"]["knee_l"][0] - self.bend_para) / 2) / \
            (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend_r = np.exp(-np.square(state_desc["joint_pos"]["knee_r"][0] - self.bend_para) / 2) / \
            (1 * np.sqrt(2 * np.pi)) - self.bend_base
        rew_bend = param.w_bend * (rew_bend_l + rew_bend_r )

        rew_all = {'straight': rew_straight, 'bend': rew_bend}
        return rew_all

    def v7(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_speed_fix = self.target_vel[0] ** 2 + self.target_vel[2] ** 2
        rew_speed_fix -= (state_desc["body_vel"]["pelvis"][0] - self.target_vel[0]) ** 2 + \
                         (state_desc["body_vel"]["pelvis"][2] - self.target_vel[2]) ** 2
        rew_speed_fix *= param.w_speed
        rew_speed_fix += 18

        rew_bend = param.w_bend * (
                max( min( -state_desc["joint_pos"]["knee_l"][0] ,
                     state_desc["joint_pos"]["knee_l"][0] - 2 * self.bend_para ), 0. ) + 
                max( min( -state_desc["joint_pos"]["knee_r"][0] ,
                     state_desc["joint_pos"]["knee_r"][0] - 2 * self.bend_para ), 0. ))

        rew_all = {'straight': rew_straight, 'bend': rew_bend, 'speed': rew_speed_fix}
        return rew_all

    def v8(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_speed_fix = self.target_vel[0] ** 2 + self.target_vel[2] ** 2
        rew_speed_fix -= (state_desc["body_vel"]["pelvis"][0] - self.target_vel[0]) ** 2 + \
                         (state_desc["body_vel"]["pelvis"][2] - self.target_vel[2]) ** 2
        rew_speed_fix *= param.w_speed
        rew_speed_fix += 16

        rew_bend = param.w_bend * (
                max( min( -state_desc["joint_pos"]["knee_l"][0] ,
                     state_desc["joint_pos"]["knee_l"][0] - 2 * self.bend_para ), 0. ) + 
                max( min( -state_desc["joint_pos"]["knee_r"][0] ,
                     state_desc["joint_pos"]["knee_r"][0] - 2 * self.bend_para ), 0. ))

        rew_all = {'straight': rew_straight, 'bend': rew_bend, 'speed': rew_speed_fix}
        return rew_all

    def v9(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_speed_fix = self.target_vel[0] ** 2 + self.target_vel[2] ** 2
        rew_speed_fix -= (state_desc["misc"]["mass_center_vel"][0] - self.target_vel[0]) ** 2 + \
                         (state_desc["misc"]["mass_center_vel"][2] - self.target_vel[2]) ** 2
        rew_speed_fix *= param.w_speed
        rew_speed_fix += 14

        rew_bend = param.w_bend * (
                max( min( -state_desc["joint_pos"]["knee_l"][0] ,
                     state_desc["joint_pos"]["knee_l"][0] - 2 * self.bend_para ), 0. ) + 
                max( min( -state_desc["joint_pos"]["knee_r"][0] ,
                     state_desc["joint_pos"]["knee_r"][0] - 2 * self.bend_para ), 0. ))

        rew_all = {'straight': rew_straight, 'bend': rew_bend, 'speed': rew_speed_fix}
        return rew_all

    def v10(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_speed = param.w_speed * (
             np.maximum(self.target_vel[0] ** 2 - (self.target_vel[0] - state_desc["body_vel"]["pelvis"][0]) ** 2, 0) +
             np.maximum(self.target_vel[2] ** 2 - (self.target_vel[2] - state_desc["body_vel"]["pelvis"][2]) ** 2, 0))

        rew_all = {'straight': rew_straight, 'speed': rew_speed}
        return rew_all

    def v11(self, state_desc):

        rew_all = self.v10(state_desc)

        rew_bend = param.w_bend * (
            max(min(-state_desc['joint_pos']['knee_l'][0], state_desc['joint_pos']['knee_l'][0] - 2 * self.bend_para), 0) +
            max(min(-state_desc['joint_pos']['knee_r'][0], state_desc['joint_pos']['knee_r'][0] - 2 * self.bend_para), 0))

        rew_all['bend'] = rew_bend
        return rew_all

    def v12(self, state_desc):

        rew_straight = self._rew_straight_v1(state_desc)

        rew_speed = param.w_speed * (
             np.maximum(np.exp(np.abs(self.target_vel[0])) - 
                np.exp(np.abs(self.target_vel[0] - state_desc["body_vel"]["pelvis"][0])), 0) +
             np.maximum(np.exp(np.abs(self.target_vel[2])) - 
                np.exp(np.abs(self.target_vel[2] - state_desc["body_vel"]["pelvis"][2])), 0))

        rew_bend = -param.w_bend * (
            np.clip(state_desc["joint_pos"]["knee_l"][0], self.bend_para, 0) + 
            np.clip(state_desc["joint_pos"]["knee_r"][0], self.bend_para, 0))

        rew_all = {'straight': rew_straight, 'speed': rew_speed, 'bend': rew_bend}
        return rew_all

    def v13(self, state_desc):

        rew_straight = self._rew_straight_v2(state_desc, shift_base=0.0835)

        rew_speed = param.w_speed * (
             np.maximum(np.sqrt(np.abs(self.target_vel[0])) - 
                np.sqrt(np.abs(self.target_vel[0] - state_desc["body_vel"]["pelvis"][0])), 0) +
             np.maximum(np.sqrt(np.abs(self.target_vel[2])) - 
                np.sqrt(np.abs(self.target_vel[2] - state_desc["body_vel"]["pelvis"][2])), 0))

        rew_bend = -param.w_bend * (
            np.clip(state_desc["joint_pos"]["knee_l"][0], self.bend_para, 0) + 
            np.clip(state_desc["joint_pos"]["knee_r"][0], self.bend_para, 0))

        rew_all = {'straight': rew_straight, 'speed': rew_speed, 'bend': rew_bend}
        return rew_all