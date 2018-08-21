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

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res += cm_pos + state_desc["misc"]["mass_center_vel"]

    return res