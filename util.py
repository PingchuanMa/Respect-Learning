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
