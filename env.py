from osim import env
import param

class ProstheticsEnv(env.ProstheticsEnv):
    
    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        rew_speed = param.w_speed * (9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2)
        rew_pose = param.w_pose * (state_desc["body_pos"]["pelvis"][0] < state_desc["body_pos"]["head"][0] and 
            state_desc["body_pos"]["pelvis"][1] < state_desc["body_pos"]["head"][1])
        rew_total = rew_speed + rew_pose
        return rew_total
