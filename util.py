import copy
import numpy as np

def state_desc_to_ob(state_desc, difficulty, mirror=False, no_acc=False, fix_target=False, current_target_vel=None, 
    target_tau=0, target_vx=1.25):
    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    if mirror:
        if no_acc:
            info_types = ["body_pos", "body_pos_rot", "body_vel", "body_vel_rot"]
        else:
            info_types = ["body_pos", "body_pos_rot", "body_vel", "body_vel_rot", "body_acc", "body_acc_rot"]
        for body_part in ["toes_l", "talus_l", "calcn_l", "tibia_l", "pros_foot_r", "pros_tibia_r"]:
            mirror_name = (body_part[:-1] + "r" ) if body_part[-1] == 'l' else (body_part[:-1] + "l" )
            for info_type in info_types:
                state_desc[info_type][mirror_name] = [ 0 ] * len(state_desc[info_type][body_part])
        
        for muscle in ['gastroc_l', 'soleus_l', 'tib_ant_l']:
            mirror_name = muscle[:-1] + "r" 
            state_desc['muscles'][mirror_name] = copy.deepcopy(state_desc["muscles"][muscle])
            for item in state_desc['muscles'][mirror_name]:
                state_desc['muscles'][mirror_name][item] = 0
        
        if "pros_foot_r_0" in state_desc["forces"]:
            state_desc["forces"]["foot_r"] = copy.deepcopy( state_desc["forces"]['pros_foot_r_0'] )
            del state_desc["forces"]['pros_foot_r_0']

            state_desc["forces"]["foot_l"][12:18] = list(map( lambda x: x[0] - x[1], zip( state_desc["forces"]["foot_l"][12:18] , state_desc["forces"]["foot_l"][18:] ) ) ) 
            state_desc["forces"]["foot_l"] = state_desc["forces"]["foot_l"][:18]

        for force in ['gastroc_l', 'soleus_l', 'tib_ant_l']:
            mirror_name = force[:-1] + "r" 
            state_desc['forces'][mirror_name] = [0]

    if mirror:
        body_list = ["pelvis", "head", "torso", "toes_l", "toes_r",
                     "talus_l", "talus_r", "calcn_l", "calcn_r",
                     "tibia_l", "tibia_r", "femur_l", "femur_r",
                     "pros_foot_l", "pros_foot_r", "pros_tibia_l",
                     "pros_tibia_r"]
    else:
        body_list = ["pelvis", "head", "torso", "toes_l", "talus_l",
                     "calcn_l", "tibia_l", "femur_l", "femur_r",
                     "pros_foot_r", "pros_tibia_r"]

    target_vel = [0, 0, 0]

    if difficulty > 0:
        # target vel (veltical is meaningless)
        # res += state_desc["target_vel"][0::2]
        if current_target_vel is None:
            target_vel = state_desc["target_vel"]
        else:
            target_vel = current_target_vel

    if fix_target:
        target_vel = [target_vx, 0, 0]

    if no_acc:
        info_types = ["body_pos", "body_pos_rot", "body_vel", "body_vel_rot"]
    else:
        info_types = ["body_pos", "body_pos_rot", "body_vel", "body_vel_rot", "body_acc", "body_acc_rot"]
    for body_part in body_list:
        cur = []
        for info_type in info_types:
            cur += state_desc[info_type][body_part]
        if body_part == "pelvis":
            pelvis = copy.deepcopy(cur)
            cur[0:3] = [0]*3
            for i in range(3):
                cur[i + 6] -= target_vel[i]
            res += cur
        else:
            for i in range(6):
                cur[i] -= pelvis[i]
            res += cur  # manual bug fix for official repo

    if no_acc:
        info_types = ["joint_pos", "joint_vel"]
    else:
        info_types = ["joint_pos", "joint_vel", "joint_acc"]
    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r", "ground_pelvis"]:
        for info_type in info_types:
            res += state_desc[info_type][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        for info_type in ["activation", "fiber_force", "fiber_length", "fiber_velocity"]:
            res += [state_desc["muscles"][muscle][info_type]]

    for force in sorted(state_desc["forces"].keys()):
        res += state_desc["forces"][force]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    cm_vel = [state_desc["misc"]["mass_center_vel"][i] - target_vel[i] for i in range(3)]
    if no_acc:
        res += cm_pos + cm_vel
    else:
        res += cm_pos + cm_vel + state_desc["misc"]["mass_center_acc"]
    return np.array(res)


def cascade_helper(state_desc, difficulty, mirror=False):
    # Augmented environment from the L2R challenge
    res = []
    end_points = dict()
    pelvis = None

    if mirror:
        for body_part in ["toes_l", "talus_l", "calcn_l", "tibia_l", "pros_foot_r", "pros_tibia_r"]:
            mirror_name = (body_part[:-1] + "r") if body_part[-1] == 'l' else (body_part[:-1] + "l")
            for info_type in ["body_pos", "body_vel", "body_pos_rot", "body_vel_rot"]:
                state_desc[info_type][mirror_name] = [0] * len(state_desc[info_type][body_part])

        for muscle in ['gastroc_l', 'soleus_l', 'tib_ant_l']:
            mirror_name = muscle[:-1] + "r"
            state_desc['muscles'][mirror_name] = copy.deepcopy(state_desc["muscles"][muscle])
            for item in state_desc['muscles'][mirror_name]:
                state_desc['muscles'][mirror_name][item] = 0

        if "pros_foot_r_0" in state_desc["forces"]:
            state_desc["forces"]["foot_r"] = copy.deepcopy(state_desc["forces"]['pros_foot_r_0'])
            del state_desc["forces"]['pros_foot_r_0']

            state_desc["forces"]["foot_l"][12:18] = list(map(lambda x: x[0] - x[1],
                                                             zip(state_desc["forces"]["foot_l"][12:18],
                                                                 state_desc["forces"]["foot_l"][18:])))
            state_desc["forces"]["foot_l"] = state_desc["forces"]["foot_l"][:18]

        for force in ['gastroc_l', 'soleus_l', 'tib_ant_l']:
            mirror_name = force[:-1] + "r"
            state_desc['forces'][mirror_name] = [0]

    if mirror:
        body_list = ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r", "calcn_l", "calcn_r", \
                     "tibia_l", "tibia_r", "femur_l", "femur_r", "pros_foot_l", "pros_foot_r", "pros_tibia_l",
                     "pros_tibia_r"]
    else:
        body_list = ["pelvis", "head", "torso", "toes_l", "talus_l", "calcn_l", "tibia_l", "femur_l", "femur_r",
                     "pros_foot_r", "pros_tibia_r"]

    target_vel = [0, 0, 0]

    if difficulty > 0:
        # target vel (veltical is meaningless)
        # res += state_desc["target_vel"][0::2]
        target_vel = state_desc["target_vel"]

    start = len(res)

    for body_part in body_list:
        cur = []
        for info_type in ["body_pos", "body_pos_rot", "body_vel", "body_vel_rot", "body_acc", "body_acc_rot"]:
            cur += state_desc[info_type][body_part]
        if body_part == "pelvis":
            pelvis = copy.deepcopy(cur)
            for i in range(3):
                cur[i + 6] -= target_vel[i]
            res += cur
        else:
            for i in range(6):
                cur[i] -= pelvis[i]
            res += cur  # manual bug fix for official repo

    end = len(res)
    end_points['body'] = (start, end)
    start = end

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r", "ground_pelvis"]:
        for info_type in ["joint_pos", "joint_vel", "joint_acc"]:
            res += state_desc[info_type][joint]
            end = len(res)
            end_points[info_type] = (start, end)
            start = end

    for muscle in sorted(state_desc["muscles"].keys()):
        for info_type in ["activation", "fiber_force", "fiber_length", "fiber_velocity"]:
            res += [state_desc["muscles"][muscle][info_type]]

    end = len(res)
    end_points['muscle'] = (start, end)
    start = end

    for force in sorted(state_desc["forces"].keys()):
        res += state_desc["forces"][force]

    end = len(res)
    end_points['force'] = (start, end)
    start = end

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    cm_vel = [state_desc["misc"]["mass_center_vel"][i] - target_vel[i] for i in range(3)]
    res += cm_pos + cm_vel + state_desc["misc"]["mass_center_acc"]

    end = len(res)
    end_points['misc'] = (start, end)
    return end_points

def state_desc_to_ob_idx(state_desc, difficulty, no_acc=False):
    # Augmented environment from the L2R challenge

    idx_dict = {}
    shift_factor = []
    mirror_obs_idx = []

    append_list = lambda x ,y: list(range( x, x + y ))

    # if difficulty > 0:
    #     # target vel (veltical is meaningless)
    #     shift_factor += [ 1, 1 ]
    #     mirror_obs_idx += append_list( len(mirror_obs_idx) , 2 )

    idx_dict["body_part"] = {}

    # "pelvis"
    c = 4 if no_acc else 6

    shift_factor += [1, 1, 1] * c
    idx_dict["body_part"]["pelvis"] = len(mirror_obs_idx)
    mirror_obs_idx += append_list( len(mirror_obs_idx) , 3 * c )

    body_list = [ "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r", "calcn_l", "calcn_r", \
        "tibia_l", "tibia_r", "femur_l", "femur_r", "pros_foot_l", "pros_foot_r", "pros_tibia_l", "pros_tibia_r"]

    c = 4 if no_acc else 6
    for body_part in body_list:
        # ["body_pos", "body_vel", "body_pos_rot", "body_vel_rot"] total 4 * 3 items
        if "_r" in body_part or "_l" in body_part:
            shift_factor += [1, 1, -1] * c
        else:
            shift_factor += [1, 1, 1] * c

        idx_dict["body_part"][body_part] = len(mirror_obs_idx)
        mirror_obs_idx += append_list( len(mirror_obs_idx) , 3*c )
        
        if "_r" in body_part and body_part[:-1]+"l" in idx_dict["body_part"]:
            mirror_part_idx = idx_dict["body_part"][body_part[:-1]+"l"]
            mirror_obs_idx[mirror_part_idx: mirror_part_idx + 12 ] , mirror_obs_idx[ -12 :  ] = \
                mirror_obs_idx[ -12 :  ] , mirror_obs_idx[mirror_part_idx: mirror_part_idx + 12 ]

    idx_dict["joints"] = {}

    c = 2 if no_acc else 3

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
        
        num = len(state_desc['joint_pos'][joint]) * 3
        shift_factor += [1] * num

        idx_dict["joints"][joint] = len(mirror_obs_idx)
        mirror_obs_idx += append_list( len(mirror_obs_idx) , num )
        
        if "_r" in joint and joint[:-1]+"l" in idx_dict["joints"]:
            mirror_part_idx = idx_dict["joints"][joint[:-1]+"l"]
            mirror_obs_idx[mirror_part_idx: mirror_part_idx + num ] , mirror_obs_idx[ -num :  ] = \
                mirror_obs_idx[ -num :  ] , mirror_obs_idx[mirror_part_idx: mirror_part_idx + num ]

    # special case for "ground_pelvis" 6 * 2 items
    shift_factor += [1 , 1, -1] * 2 * c
    idx_dict["joints"]["ground_pelvis"] = len(mirror_obs_idx)
    mirror_obs_idx += append_list( len(mirror_obs_idx) , 3 * 2 * c )

    idx_dict["muscles"] = {}
    for muscle in sorted(state_desc["muscles"].keys()):
        # ["activation", "fiber_force", "fiber_length", "fiber_velocity"] 4
        shift_factor += [1, 1, 1, 1]

        idx_dict["muscles"][muscle] = len(mirror_obs_idx)
        mirror_obs_idx += append_list( len(mirror_obs_idx) , 4 )
        
        if "_r" in muscle and muscle[:-1]+"l" in idx_dict["muscles"]:
            mirror_part_idx = idx_dict["muscles"][muscle[:-1]+"l"]
            mirror_obs_idx[mirror_part_idx: mirror_part_idx + 4 ] , mirror_obs_idx[ -4 :  ] = \
                mirror_obs_idx[ -4 :  ] , mirror_obs_idx[mirror_part_idx: mirror_part_idx + 4 ]

    # forces_list=['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r', 'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 'gastroc_l', 'soleus_l', 'tib_ant_l', 'ankleSpring', 'pros_foot_r_0', 'foot_l', 'HipLimit_r', 'HipLimit_l', 'KneeLimit_r', 'KneeLimit_l', 'AnkleLimit_r', 'AnkleLimit_l', 'HipAddLimit_r', 'HipAddLimit_l']
    idx_dict["forces"] = {}

    for force in sorted(state_desc["forces"].keys()):
        
        num = len(state_desc['forces'][force])

        shift_factor += [1] * num

        idx_dict["forces"][force] = len(mirror_obs_idx)
        mirror_obs_idx += append_list( len(mirror_obs_idx) , num )
        
        if "_r" in force and force[:-1]+"l" in idx_dict["forces"]:
            mirror_part_idx = idx_dict["forces"][ force[:-1]+"l" ]
            mirror_obs_idx[mirror_part_idx: mirror_part_idx + num ] , mirror_obs_idx[ -num :  ] = \
                mirror_obs_idx[ -num :  ] , mirror_obs_idx[mirror_part_idx: mirror_part_idx + num ]
    
    shift_factor[ idx_dict["forces"]["foot_l"] : idx_dict["forces"]["foot_l"] + 18 ] = [1, 1, -1] * 6
    shift_factor[ idx_dict["forces"]["foot_r"] : idx_dict["forces"]["foot_r"] + 18 ] = [1, 1, -1] * 6
        
    c = 2 if no_acc else 3
    # for misc info
    shift_factor += [1] * 3*c
    mirror_obs_idx += append_list( len(mirror_obs_idx) , 3*c )

    return mirror_obs_idx, shift_factor

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


def get_mirror_id( state_desc, difficulty, no_acc ):
    # 0  ~ 7  right
    # 8  ~ 15 left
    # 16 ~ 18 left only
    act_idx = [ 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 16, 17, 18 ]
    ob_idx, shift_factor = state_desc_to_ob_idx( state_desc, difficulty, no_acc )
    print(len(act_idx))
    print(len(ob_idx))
    print(len(shift_factor))
    return  np.array(ob_idx), np.array(act_idx), np.array(shift_factor)
