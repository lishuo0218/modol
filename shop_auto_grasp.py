import time
from pathlib import Path

import numpy as np

import mujoco


SHOP_XML = Path(__file__).with_name("shop.xml")

START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
]

LEFT_JOINTS = [
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
    "vx300s_left/left_finger",
    "vx300s_left/right_finger",
]

RIGHT_JOINTS = [
    "vx300s_right/waist",
    "vx300s_right/shoulder",
    "vx300s_right/elbow",
    "vx300s_right/forearm_roll",
    "vx300s_right/wrist_angle",
    "vx300s_right/wrist_rotate",
    "vx300s_right/left_finger",
    "vx300s_right/right_finger",
]


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def set_gripper(model: mujoco.MjModel, data: mujoco.MjData, side: str, left: float, right: float):
    if side not in {"left", "right"}:
        raise ValueError(side)
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vx300s_{side}_left_finger")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vx300s_{side}_right_finger")
    data.ctrl[left_act] = left
    data.ctrl[right_act] = right


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_adr = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_adr] = value


def set_joint_ctrl(model: mujoco.MjModel, data: mujoco.MjData, actuator_name: str, value: float):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    data.ctrl[actuator_id] = value


def solve_ik_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    joint_names: list[str],
    target_pos: np.ndarray,
    max_iters: int = 220,
    damping: float = 0.01,
    step_scale: float = 1.2,
    tol: float = 0.005,
):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_names]
    dof_addrs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
    qpos_addrs = [int(model.jnt_qposadr[jid]) for jid in joint_ids]

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        ee = np.array(data.site_xpos[site_id], dtype=np.float64)
        err = target_pos - ee
        if float(np.linalg.norm(err)) < tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j = jacp[:, dof_addrs]
        jj_t = j @ j.T + (damping**2) * np.eye(3)
        dq = j.T @ np.linalg.solve(jj_t, err * step_scale)

        for k, jid in enumerate(joint_ids):
            qadr = qpos_addrs[k]
            data.qpos[qadr] = data.qpos[qadr] + dq[k]
            lo, hi = model.jnt_range[jid]
            if lo < hi:
                data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))
            data.qvel[dof_addrs[k]] = 0.0


def activate_grasp_weld(model: mujoco.MjModel, data: mujoco.MjData):
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grab_box")
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vx300s_left/gripper_link")
    box_pos = np.array(data.xpos[box_body_id], dtype=np.float64)
    box_quat = np.array(data.xquat[box_body_id], dtype=np.float64)
    grip_pos = np.array(data.xpos[gripper_body_id], dtype=np.float64)
    grip_quat = np.array(data.xquat[gripper_body_id], dtype=np.float64)

    inv_pos = np.zeros(3, dtype=np.float64)
    inv_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_negPose(inv_pos, inv_quat, grip_pos, grip_quat)

    rel_pos = np.zeros(3, dtype=np.float64)
    rel_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulPose(rel_pos, rel_quat, inv_pos, inv_quat, box_pos, box_quat)
    return rel_pos, rel_quat


def apply_attachment(model: mujoco.MjModel, data: mujoco.MjData, rel_pos: np.ndarray, rel_quat: np.ndarray):
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "grab_box_joint")
    box_qpos_adr = int(model.jnt_qposadr[box_joint_id])
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vx300s_left/gripper_link")

    grip_pos = np.array(data.xpos[gripper_body_id], dtype=np.float64)
    grip_quat = np.array(data.xquat[gripper_body_id], dtype=np.float64)

    out_pos = np.zeros(3, dtype=np.float64)
    out_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulPose(out_pos, out_quat, grip_pos, grip_quat, rel_pos, rel_quat)

    data.qpos[box_qpos_adr : box_qpos_adr + 3] = out_pos
    data.qpos[box_qpos_adr + 3 : box_qpos_adr + 7] = out_quat


def main():
    if not SHOP_XML.exists():
        raise SystemExit(f"Missing {SHOP_XML}")

    model = mujoco.MjModel.from_xml_path(str(SHOP_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.eq_active[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")] = 0

    for name, value in zip(LEFT_JOINTS, START_ARM_POSE[:8], strict=True):
        set_joint_qpos(model, data, name, value)
    for name, value in zip(RIGHT_JOINTS, START_ARM_POSE[8:], strict=True):
        set_joint_qpos(model, data, name, value)

    mujoco.mj_forward(model, data)

    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grab_box")
    belt_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "conveyor_belt_1")
    belt_pos = np.array(data.geom_xpos[belt_geom_id], dtype=np.float64)
    belt_half = np.array(model.geom_size[belt_geom_id], dtype=np.float64)
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "grab_box_joint")
    box_qpos_adr = int(model.jnt_qposadr[box_joint_id])
    box_dof_adr = int(model.jnt_dofadr[box_joint_id])
    box_half_size = 0.02
    data.qpos[box_qpos_adr : box_qpos_adr + 3] = belt_pos + np.array([0.0, 0.0, belt_half[2] + box_half_size + 0.002])
    data.qpos[box_qpos_adr + 3 : box_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[box_dof_adr : box_dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    box_pos = np.array(data.xpos[box_body_id], dtype=np.float64)
    set_gripper(model, data, "left", left=0.057, right=-0.057)
    set_gripper(model, data, "right", left=0.057, right=-0.057)

    for j in LEFT_JOINTS[:6]:
        q = float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])])
        set_joint_ctrl(model, data, "vx300s_left_" + j.split("/")[-1], q)
    for j in RIGHT_JOINTS[:6]:
        q = float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])])
        set_joint_ctrl(model, data, "vx300s_right_" + j.split("/")[-1], q)

    for _ in range(50):
        mujoco.mj_step(model, data)

    start_box_z = float(data.xpos[box_body_id][2])
    box_pos = np.array(data.xpos[box_body_id], dtype=np.float64)

    grasp_offset = np.array([0.0, 0.0, 0.02])
    above = box_pos + grasp_offset + np.array([0.0, 0.0, 0.22])
    grasp = box_pos + grasp_offset
    lift = box_pos + grasp_offset + np.array([0.0, 0.0, 0.30])

    left_arm = LEFT_JOINTS[:6]
    solve_ik_position(model, data, "cali_left_site1", left_arm, above)
    for j in left_arm:
        q = float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])])
        set_joint_ctrl(model, data, "vx300s_left_" + j.split("/")[-1], q)
    for _ in range(200):
        mujoco.mj_step(model, data)

    solve_ik_position(model, data, "cali_left_site1", left_arm, grasp)
    for j in left_arm:
        q = float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])])
        set_joint_ctrl(model, data, "vx300s_left_" + j.split("/")[-1], q)
    for _ in range(200):
        mujoco.mj_step(model, data)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cali_left_site1")
    ee_pos = np.array(data.site_xpos[ee_site_id], dtype=np.float64)
    box_pos_now = np.array(data.xpos[box_body_id], dtype=np.float64)
    print(f"ee_pos={ee_pos.round(3)} box_pos={box_pos_now.round(3)} dist={float(np.linalg.norm(ee_pos-box_pos_now)):.3f}")
    rel_pos, rel_quat = activate_grasp_weld(model, data)

    for _ in range(50):
        set_gripper(model, data, "left", left=0.021, right=-0.021)
        mujoco.mj_step(model, data)

    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = data.qpos
    data_ik.qvel[:] = 0.0
    data_ik.ctrl[:] = data.ctrl
    mujoco.mj_forward(model, data_ik)
    solve_ik_position(model, data_ik, "cali_left_site1", left_arm, lift)
    for j in left_arm:
        q = float(data_ik.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])])
        set_joint_ctrl(model, data, "vx300s_left_" + j.split("/")[-1], q)
    for _ in range(400):
        mujoco.mj_step(model, data)
        apply_attachment(model, data, rel_pos, rel_quat)
        mujoco.mj_forward(model, data)

    end_box_z = float(data.xpos[box_body_id][2])
    lifted = end_box_z - start_box_z
    print(f"grab_box dz={lifted:.3f}m (start={start_box_z:.3f}, end={end_box_z:.3f})")
    if lifted > 0.05:
        print("SUCCESS: object lifted")
    else:
        print("NOT_SURE: object not clearly lifted (try adjusting offsets)")

    time.sleep(0.2)


if __name__ == "__main__":
    main()
