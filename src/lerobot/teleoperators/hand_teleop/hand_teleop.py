#!/usr/bin/env python

import logging
from functools import cached_property
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.robot_arm_viewer import resolve_viewer_urdf_path

from ..teleoperator import Teleoperator
from .config_hand_teleop import HandTeleopConfig

logger = logging.getLogger(__name__)

SO_ARM_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
GEM_ARM_ACTION_NAMES = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "joint_7.pos",
    "gripper.pos",
)


class HandTeleop(Teleoperator):
    """LeRobot teleoperator that uses hand-teleop webcam tracking as a leader arm."""

    config_class = HandTeleopConfig
    name = "hand_teleop"

    def __init__(self, config: HandTeleopConfig):
        super().__init__(config)
        self.config = config
        self._tracker = None
        self._is_connected = False
        self._base_joints: dict[str, np.ndarray] = {}

    @cached_property
    def action_features(self) -> dict[str, type]:
        action_names = self._action_names()
        if self.config.hand == "both":
            return {
                **{f"left_{name}": float for name in action_names},
                **{f"right_{name}": float for name in action_names},
            }
        return dict.fromkeys(action_names, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        _ = calibrate
        safe_range = self.config.safe_range
        urdf_path = self._resolve_urdf_path()
        if self.config.hand == "both":
            from hand_teleop.tracking.dual_tracker import DualHandTracker

            self._tracker = DualHandTracker(
                cam_idx=self.config.cam_idx,
                device=self.config.device,
                model=self.config.model,
                show_viz=self.config.show_viz,
                focal_ratio=self.config.focal_ratio,
                urdf_path=urdf_path,
                frame_name=self.config.frame_name,
                safe_range=safe_range,
                debug_mode=self.config.debug_mode,
                kf_dt=1 / self.config.fps,
                kf_q=self.config.kf_q,
                kf_r=self.config.kf_r,
                start_paused=self.config.start_paused,
            )
            self._base_joints = {
                "left": self._resolve_base_joint(self.config.left_base_joint),
                "right": self._resolve_base_joint(self.config.right_base_joint),
            }
        else:
            from hand_teleop.tracking.tracker import HandTracker

            self._tracker = HandTracker(
                cam_idx=self.config.cam_idx,
                device=self.config.device,
                model=self.config.model,
                hand=self.config.hand,
                show_viz=self.config.show_viz,
                focal_ratio=self.config.focal_ratio,
                urdf_path=urdf_path,
                frame_name=self.config.frame_name,
                safe_range=safe_range,
                use_scroll=self.config.use_scroll,
                kf_dt=1 / self.config.fps,
                kf_q=self.config.kf_q,
                kf_r=self.config.kf_r,
                debug_mode=self.config.debug_mode,
            )
            if not self.config.start_paused:
                self._tracker._resume()
            self._base_joints = {self.config.hand: self._resolve_base_joint(self._configured_base_for_hand())}

        self._is_connected = True
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _resolve_urdf_path(self) -> str:
        if self.config.urdf_path.lower() in {"gem", "gem_follower", "bi_gem", "bi_gem_follower"}:
            urdf_path = resolve_viewer_urdf_path("gem")
            if not urdf_path.is_file():
                raise FileNotFoundError(
                    f"GEM viewer URDF not found at {urdf_path}. Set --teleop.urdf_path to an explicit "
                    ".urdf path or install the robot-arm-viewer assets."
                )
            return str(urdf_path)
        return self.config.urdf_path

    def _action_names(self) -> tuple[str, ...]:
        if self.config.urdf_path.lower() in {"gem", "gem_follower", "bi_gem", "bi_gem_follower"}:
            return GEM_ARM_ACTION_NAMES
        return SO_ARM_ACTION_NAMES

    def _configured_base_for_hand(self) -> tuple[float, ...] | None:
        return self.config.left_base_joint if self.config.hand == "left" else self.config.right_base_joint

    def _resolve_base_joint(
        self, configured: tuple[float, ...] | None
    ) -> np.ndarray:
        if configured is not None:
            return np.asarray(configured, dtype=np.float32)
        return self._default_base_joint()

    def _default_base_joint(self) -> np.ndarray:
        if self._tracker is None or self._tracker.robot_kin is None:
            return np.zeros(len(self._action_names()), dtype=np.float32)

        follower_pos = np.array([0.2, 0, 0.1])
        follower_rot = R.from_euler("ZYX", [0, 45, -90], degrees=True).as_matrix()
        target = np.eye(4)
        target[:3, :3] = follower_rot
        target[:3, 3] = follower_pos
        q0 = np.zeros(self._tracker.robot_kin.nq)
        q0[: min(3, len(q0))] = [0, 2, 2][: min(3, len(q0))]
        arm = np.degrees(self._tracker.robot_kin.ik(q0, target)[: self._tracker.robot_kin.nq])
        return np.append(arm, 5.0).astype(np.float32)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self._tracker is None:
            raise RuntimeError("HandTeleop is connected but tracker was not initialized.")

        if self.config.hand == "both":
            action: RobotAction = {}
            for hand in ("left", "right"):
                joints = self._tracker.read_hand_state_joint(hand, self._base_joints[hand])
                hand_action = _joints_to_action(joints, self._action_names())
                action.update({f"{hand}_{key}": value for key, value in hand_action.items()})
            return action

        joints = self._tracker.read_hand_state_joint(self._base_joints[self.config.hand])
        return _joints_to_action(joints, self._action_names())

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        _ = feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
        self._tracker = None
        self._is_connected = False
        logger.info("%s disconnected.", self)


def _joints_to_action(joints: np.ndarray, action_names: tuple[str, ...]) -> RobotAction:
    return {name: float(value) for name, value in zip(action_names, joints, strict=True)}
