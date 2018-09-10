def state_desc_to_ob(state_desc):
    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head", "torso", "toes_l", "talus_l", "calcn_l", "tibia_l", "femur_l", "femur_r", "pros_foot_r", "pros_tibia_r"]:
        cur = []
        cur += state_desc["body_pos"][body_part]
        cur += state_desc["body_vel"][body_part]
        cur += state_desc["body_acc"][body_part]
        cur += state_desc["body_pos_rot"][body_part]
        cur += state_desc["body_vel_rot"][body_part]
        cur += state_desc["body_acc_rot"][body_part]
        if body_part == "pelvis":
            pelvis = cur
            res += cur
        else:
            cur_upd = cur
            cur_upd[:3] = [cur[i] - pelvis[i] for i in range(3)]
            cur_upd[9:12] = [cur[i] - pelvis[i] for i in range(9, 12)]
            res += cur_upd  # manual bug fix for official repo

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r", "ground_pelvis"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_force"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    res += cm_pos + state_desc["misc"]["mass_center_vel"]

    return res

def state_desc_to_ob_idx(state_desc):
    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head", "torso", "toes_l", "talus_l", "calcn_l", "tibia_l", "femur_l", "femur_r", "pros_foot_r", "pros_tibia_r"]:
        cur = []
        cur += state_desc["body_pos"][body_part]
        cur += state_desc["body_vel"][body_part]
        cur += state_desc["body_acc"][body_part]
        cur += state_desc["body_pos_rot"][body_part]
        cur += state_desc["body_vel_rot"][body_part]
        cur += state_desc["body_acc_rot"][body_part]
        if body_part == "pelvis":
            pelvis = cur
            res += cur
        else:
            cur_upd = cur
            cur_upd[:3] = [cur[i] - pelvis[i] for i in range(3)]
            cur_upd[9:12] = [cur[i] - pelvis[i] for i in range(9, 12)]
            res += cur_upd  # manual bug fix for official repo

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r", "ground_pelvis"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_force"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    res += cm_pos + state_desc["misc"]["mass_center_vel"]

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