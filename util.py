import copy

def state_desc_to_ob(state_desc):
    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head", "torso", "toes_l", "talus_l", "calcn_l", "tibia_l", "femur_l", "femur_r", "pros_foot_r", "pros_tibia_r"]:
        cur = []
        for info_type in ["body_pos", "body_vel", "body_pos_rot", "body_vel_rot"]:
            cur += state_desc[info_type][body_part]
        if body_part == "pelvis":
            pelvis = cur
            res += cur
        else:
            cur_upd = cur
            cur_upd[:3] = [cur[i] - pelvis[i] for i in range(3)]
            cur_upd[6:9] = [cur[i] - pelvis[i] for i in range(6, 9)]
            res += cur_upd  # manual bug fix for official repo

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r", "ground_pelvis"]:
        for info_type in ["joint_pos", "joint_vel"]:
            res += state_desc["joint_pos"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        for info_type in ["activation", "fiber_force", "fiber_length", "fiber_velocity"]:
            res += [state_desc["muscles"][muscle][info_type]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    res += cm_pos + state_desc["misc"]["mass_center_vel"]

    return res

def state_desc_to_ob_idx(state_desc):
    # Augmented environment from the L2R challenge

    res = []
    pelvis = None

    idx_dict = copy.deepcopy(state_desc)
    shift_factor = []
    mirror_obs_idx = []

    append_list = lambda x ,y: list(range( x, y ))

    for body_part in ["pelvis", "head", "torso", "toes_l", "talus_l", "calcn_l", "tibia_l", "femur_l", "femur_r", "pros_foot_r", "pros_tibia_r"]:

        if "_r" in body_part or "_l" in body_part:
            cur_shift_factor = [1, 1, -1]
        else:
            cur_shift_factor = [1, 1, 1]

        for info_type in ["body_pos", "body_vel", "body_pos_rot", "body_vel_rot"]:
            shift_factor += cur_shift_factor
            idx_dict["body_pos"][body_part] = append_list( len(mirror_obs_idx), 3 )
            mirror_obs_idx += append_list( len(mirror_obs_idx) , 3 )

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r", "ground_pelvis"]:
        
        if "_r" in body_part or "_l" in body_part:
            cur_shift_factor += [1, 1, -1]
        else:
            cur_shift_factor += [1, 1, 1]

        for info_type in ["joint_pos", "joint_vel"]:
            shift_factor += cur_shift_factor
            mirror_obs_idx += append_list( len(mirror_obs_idx) , 3 )
            idx_dict[info_type][joint] = append_list( len(mirror_obs_idx), 3 )

    for muscle in sorted(state_desc["muscles"].keys()):
        for info_type in ["activation", "fiber_force", "fiber_length", "fiber_velocity"]:
            res += [state_desc["muscles"][muscle][info_type]]
            idx_dict["muscles"][muscle][info_type]

    shift_factor += [1] * 6
    mirror_obs_idx += append_list( len(mirror_obs_idx) , 6 )

    return res

"""
0   abd_r
1   add_r
2   hamstrings_r
3   bifemsh_r
4   glut_max_r
5   iliopsoas_r
6   rect_fem_r
7   vasti_r
8   abd_l
9   add_l
10   hamstrings_l
11   bifemsh_l
12   glut_max_l
13   iliopsoas_l
14   rect_fem_l
15   vasti_l
16   gastroc_l
17   soleus_l
18   tib_ant_l
"""
def get_mirror_id( state_desc ):
    # 0  ~ 7  right
    # 8  ~ 15 left
    # 16 ~ 18 left only
    act_idx = [ 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18 ]

    ob_idx, shift_factor = state_desc_to_ob_idx( state_desc )
    return act_idx, ob_idx, shift_factor