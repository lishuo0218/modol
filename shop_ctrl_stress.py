import sys

import mujoco
import numpy as np


def main():
    model = mujoco.MjModel.from_xml_path("shop.xml")
    data = mujoco.MjData(model)

    actuator_names = [
        "vx300s_left_waist",
        "vx300s_left_shoulder",
        "vx300s_left_elbow",
        "vx300s_left_forearm_roll",
        "vx300s_left_wrist_angle",
        "vx300s_left_wrist_rotate",
        "vx300s_right_waist",
        "vx300s_right_shoulder",
        "vx300s_right_elbow",
        "vx300s_right_forearm_roll",
        "vx300s_right_wrist_angle",
        "vx300s_right_wrist_rotate",
        "vx300s_left_left_finger",
        "vx300s_left_right_finger",
        "vx300s_right_left_finger",
        "vx300s_right_right_finger",
    ]

    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
    ctrlrange = model.actuator_ctrlrange[act_ids]

    rng = np.random.default_rng(0)
    for _ in range(200):
        mujoco.mj_step(model, data)

    try:
        for _ in range(200):
            sample = rng.uniform(ctrlrange[:, 0], ctrlrange[:, 1])
            data.ctrl[act_ids] = sample
            for _ in range(50):
                mujoco.mj_step(model, data)
        print("OK: stress run completed")
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
