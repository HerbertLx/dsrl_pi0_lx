"""Smoke test for AirbotRobotEnv interface.

Run this before wiring into training:
python openpi/examples/airbot/smoke_test_robot_env.py
"""

from __future__ import annotations

import numpy as np

from airbot_env_template import AirbotConfig
from airbot_env_template import AirbotRobotEnv


def _assert_observation_schema(obs: dict) -> None:
    if "image" not in obs:
        raise AssertionError("obs missing top-level key: image")
    if "robot_state" not in obs:
        raise AssertionError("obs missing top-level key: robot_state")

    image = obs["image"]
    state = obs["robot_state"]

    for required in ["joint_positions", "gripper_position", "cartesian_position"]:
        if required not in state:
            raise AssertionError(f"robot_state missing key: {required}")

    for cam_key, cam_img in image.items():
        arr = np.asarray(cam_img)
        if arr.ndim != 3:
            raise AssertionError(f"camera {cam_key} expected 3D HWC image, got shape={arr.shape}")
        if arr.shape[2] not in (3, 4):
            raise AssertionError(f"camera {cam_key} expected 3 or 4 channels, got shape={arr.shape}")

    jp = np.asarray(state["joint_positions"])
    cp = np.asarray(state["cartesian_position"])
    gp = float(state["gripper_position"])

    if jp.shape[0] != 7:
        raise AssertionError(f"joint_positions expected dim 7, got shape={jp.shape}")
    if cp.shape[0] != 6:
        raise AssertionError(f"cartesian_position expected dim 6, got shape={cp.shape}")
    if not (0.0 <= gp <= 1.0):
        raise AssertionError(f"gripper_position expected in [0, 1], got {gp}")


def main() -> None:
    cfg = AirbotConfig(
        left_camera_id="left_cam_demo",
        right_camera_id="right_cam_demo",
        wrist_camera_id="wrist_cam_demo",
        do_reset_on_init=False,
    )
    env = AirbotRobotEnv(cfg)

    print("[检查] reset")
    env.reset()

    print("[检查] get_observation")
    obs = env.get_observation()
    _assert_observation_schema(obs)
    print("[通过] observation schema")

    print("[检查] step x 3")
    for i in range(3):
        action = np.zeros((8,), dtype=np.float32)
        action[-1] = 1.0 if i % 2 == 0 else 0.0
        step_info = env.step(action)
        obs = env.get_observation()
        _assert_observation_schema(obs)
        print(f"[通过] step={i}, info={step_info}")

    print("[完成] AirbotRobotEnv 最小接口可用")


if __name__ == "__main__":
    main()
