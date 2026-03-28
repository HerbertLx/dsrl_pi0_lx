"""Airbot RobotEnv template.

This template matches the interface expected by examples/train_utils_real.py.
Replace TODO sections with your Airbot SDK calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict

import numpy as np


@dataclass
class AirbotConfig:
    """Configuration used by AirbotRobotEnv.

    camera keys are the IDs used by your launcher/env vars and are reused in
    output image keys to keep compatibility with the current training parser.
    """

    left_camera_id: str
    right_camera_id: str
    wrist_camera_id: str
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    do_reset_on_init: bool = False


class AirbotRobotEnv:
    """Adapter that mimics droid.robot_env.RobotEnv behavior.

    Required methods for current real training pipeline:
    - reset(...)
    - step(action)
    - get_observation()
    """

    def __init__(self, config: AirbotConfig):
        self.config = config
        self.action_space = config.action_space
        self.gripper_action_space = config.gripper_action_space

        # TODO: initialize your Airbot SDK client(s), robot connection, cameras.
        self._robot = None
        self._camera = None

        if self.config.do_reset_on_init:
            self.reset()

    def reset(self, randomize: bool = False) -> None:
        """Reset robot to a safe start state.

        Args:
            randomize: Keep arg for compatibility with existing call sites.
        """
        del randomize
        # TODO: replace with your reset pipeline.
        # Example:
        # self._robot.open_gripper()
        # self._robot.move_joints(self._home_joints, blocking=True)
        return None

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """Execute one control step.

        Expected by current training code:
        - action is a 1D array, usually len=8
        - action[:7] -> arm command
        - action[7]  -> gripper command (0/1 already binarized upstream)
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 8:
            raise ValueError(f"Expected action with at least 8 dims, got {action.shape}")

        arm_cmd = action[:7]
        grip_cmd = float(action[7])

        # Safety clamp at adapter level even if caller already clips.
        arm_cmd = np.clip(arm_cmd, -1.0, 1.0)
        grip_cmd = float(np.clip(grip_cmd, 0.0, 1.0))

        # TODO: map normalized command to your hardware command API.
        # Example:
        # self._robot.send_joint_velocity(arm_cmd)
        # self._robot.set_gripper_position(grip_cmd)

        return {
            "ok": True,
            "action_space": self.action_space,
            "gripper_action_space": self.gripper_action_space,
        }

    def get_observation(self) -> Dict[str, Any]:
        """Return observation dictionary compatible with train_utils_real parser.

        Output schema:
        {
            "image": {
                "<left_camera_id>_left":  np.ndarray(H,W,3|4),
                "<right_camera_id>_left": np.ndarray(H,W,3|4),
                "<wrist_camera_id>_left": np.ndarray(H,W,3|4),
            },
            "robot_state": {
                "cartesian_position": np.ndarray(6,),
                "joint_positions": np.ndarray(7,),
                "gripper_position": float,
            },
        }
        """
        # TODO: replace with real sensor reads.
        left_img = self._read_camera(self.config.left_camera_id)
        right_img = self._read_camera(self.config.right_camera_id)
        wrist_img = self._read_camera(self.config.wrist_camera_id)

        joint_positions = self._read_joint_positions()
        gripper_position = self._read_gripper_position()
        cartesian_position = self._read_cartesian_position()

        return {
            "image": {
                f"{self.config.left_camera_id}_left": left_img,
                f"{self.config.right_camera_id}_left": right_img,
                f"{self.config.wrist_camera_id}_left": wrist_img,
            },
            "robot_state": {
                "cartesian_position": cartesian_position,
                "joint_positions": joint_positions,
                "gripper_position": float(gripper_position),
            },
        }

    # -----------------------------
    # Internal read helpers
    # -----------------------------
    def _read_camera(self, camera_id: str) -> np.ndarray:
        """Read one frame for the given camera_id.

        Must return uint8 HWC image with 3 or 4 channels.
        """
        del camera_id
        # TODO: replace with your camera SDK call.
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _read_joint_positions(self) -> np.ndarray:
        """Read 7-DoF joint state."""
        # TODO: replace with real read.
        return np.zeros((7,), dtype=np.float32)

    def _read_gripper_position(self) -> float:
        """Read normalized gripper position in [0, 1]."""
        # TODO: replace with real read.
        return 0.0

    def _read_cartesian_position(self) -> np.ndarray:
        """Read 6D end-effector pose representation."""
        # TODO: replace with real read.
        return np.zeros((6,), dtype=np.float32)


def make_airbot_env_from_envvars() -> AirbotRobotEnv:
    """Factory used by training scripts.

    Reuses the same env var names as current real launcher to minimize changes.
    """
    cfg = AirbotConfig(
        left_camera_id=_get_required_env("LEFT_CAMERA_ID"),
        right_camera_id=_get_required_env("RIGHT_CAMERA_ID"),
        wrist_camera_id=_get_required_env("WRIST_CAMERA_ID"),
    )
    return AirbotRobotEnv(cfg)


def _get_required_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
