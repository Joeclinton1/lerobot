#!/usr/bin/env python

from typing import Any

import numpy as np

from lerobot.model import RobotKinematics
from lerobot.utils.robot_arm_viewer import resolve_viewer_urdf_path

GEM_MOTOR_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
]

GEM_URDF_JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link7",
]

GEM_TARGET_FRAME = "gripper"

# Comfortable neutral posture (deg) for the 7-DOF arm. Used as a low-priority posture task so the
# redundant elbow resolves deterministically to a single, human-like configuration per target
# instead of drifting between branches. Matches the viewer home pose (all joints at zero).
GEM_NEUTRAL_POSTURE_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# robot-arm-viewer displays the GEM with +Y vertical and the arm facing -Z.
# The URDF itself is loaded under a +Z-up transform, so convert visible viewer
# control coordinates into the raw URDF frame before calling the kinematics.
GEM_VIEWER_CONTROL_TO_URDF = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
GEM_RIGHT_ARM_REFLECTION_IN_URDF = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
GEM_RIGHT_VIEWER_CONTROL_TO_URDF = GEM_RIGHT_ARM_REFLECTION_IN_URDF @ GEM_VIEWER_CONTROL_TO_URDF


class FrameAdaptedRobotKinematics:
    def __init__(self, kinematics: RobotKinematics, control_to_urdf: np.ndarray):
        self._kinematics = kinematics
        self._control_to_urdf = control_to_urdf.astype(np.float32)

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        return self._urdf_to_control(self._kinematics.forward_kinematics(joint_pos_deg))

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        return self._kinematics.inverse_kinematics(
            current_joint_pos,
            self._control_to_urdf_pose(desired_ee_pose),
            *args,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kinematics, name)

    def _urdf_to_control(self, transform: np.ndarray) -> np.ndarray:
        rotation = self._control_to_urdf
        converted = transform.copy()
        converted[:3, :3] = rotation.T @ transform[:3, :3] @ rotation
        converted[:3, 3] = rotation.T @ transform[:3, 3]
        return converted

    def _control_to_urdf_pose(self, transform: np.ndarray) -> np.ndarray:
        rotation = self._control_to_urdf
        converted = transform.copy()
        converted[:3, :3] = rotation @ transform[:3, :3] @ rotation.T
        converted[:3, 3] = rotation @ transform[:3, 3]
        return converted


def make_gem_kinematics(
    align_to_viewer: bool = True,
    arm: str = "left",
    posture_weight: float = 1e-2,
) -> RobotKinematics | FrameAdaptedRobotKinematics:
    kinematics = RobotKinematics(
        urdf_path=str(resolve_viewer_urdf_path("gem")),
        target_frame_name=GEM_TARGET_FRAME,
        joint_names=GEM_URDF_JOINT_NAMES,
        posture_target_deg=GEM_NEUTRAL_POSTURE_DEG,
        posture_weight=posture_weight,
        enforce_joint_limits=True,
    )
    if not align_to_viewer:
        return kinematics
    if arm == "left":
        return FrameAdaptedRobotKinematics(kinematics, GEM_VIEWER_CONTROL_TO_URDF)
    if arm == "right":
        return FrameAdaptedRobotKinematics(kinematics, GEM_RIGHT_VIEWER_CONTROL_TO_URDF)
    raise ValueError(f"Unsupported GEM arm {arm!r}; expected 'left' or 'right'.")
