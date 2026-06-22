#!/usr/bin/env python

import logging
from functools import cached_property
from typing import Any, Protocol

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

# Maps the viewer-aligned control frame into the GEM URDF frame for the viewer's
# default +Z-up display. This is a pure rotation, not a hand-tracking basis change.
_GEM_VIEWER_CONTROL_TO_URDF = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_GEM_RIGHT_ARM_REFLECTION_IN_URDF = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
_GEM_RIGHT_VIEWER_CONTROL_TO_URDF = _GEM_RIGHT_ARM_REFLECTION_IN_URDF @ _GEM_VIEWER_CONTROL_TO_URDF


class _RobotKinematicsLike(Protocol):
    nq: int
    urdf_path: str

    def fk(self, q: np.ndarray, frame: str | None = None) -> np.ndarray: ...

    def ik(self, q0: np.ndarray, target_t: np.ndarray, *args, **kwargs) -> np.ndarray: ...


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
        self._viewer_hand_kinematics: dict[str, _RobotKinematicsLike] | None = None

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
                debug_viz=self.config.debug_viz,
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
                debug_viz=self.config.debug_viz,
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

    def align_kinematics_to_gem_viewer(self) -> None:
        if self._tracker is None or self._tracker.robot_kin is None:
            return
        if self.config.urdf_path.lower() not in {"gem", "gem_follower", "bi_gem", "bi_gem_follower"}:
            return
        base_robot_kin = _unwrap_frame_adapted_kinematics(self._tracker.robot_kin)
        if self.config.hand == "both":
            self._viewer_hand_kinematics = {
                "left": _FrameAdaptedKinematics(
                    base_robot_kin,
                    control_to_urdf=_GEM_VIEWER_CONTROL_TO_URDF,
                ),
                "right": _FrameAdaptedKinematics(
                    base_robot_kin,
                    control_to_urdf=_GEM_RIGHT_VIEWER_CONTROL_TO_URDF,
                ),
            }
            self._reset_tracker_base_poses()
            return

        if isinstance(self._tracker.robot_kin, _FrameAdaptedKinematics):
            self._reset_tracker_base_poses()
            return
        self._tracker.robot_kin = _FrameAdaptedKinematics(
            base_robot_kin,
            control_to_urdf=_GEM_VIEWER_CONTROL_TO_URDF,
        )
        self._reset_tracker_base_poses()

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
        if self.config.urdf_path.lower() in {"gem", "gem_follower", "bi_gem", "bi_gem_follower"}:
            arm_dof = (
                self._tracker.robot_kin.nq
                if self._tracker is not None and self._tracker.robot_kin is not None
                else 7
            )
            return np.append(np.zeros(arm_dof, dtype=np.float32), 5.0).astype(np.float32)

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
    def get_action(self, current_state: RobotAction | None = None) -> RobotAction:
        if self._tracker is None:
            raise RuntimeError("HandTeleop is connected but tracker was not initialized.")

        if self.config.hand == "both":
            action: RobotAction = {}
            for hand in ("left", "right"):
                base_joints = self._base_joints_from_state(hand, current_state)
                joints = self._read_dual_hand_state_joint(hand, base_joints)
                hand_action = _joints_to_action(joints, self._action_names())
                action.update({f"{hand}_{key}": value for key, value in hand_action.items()})
            return action

        base_joints = self._base_joints_from_state(self.config.hand, current_state)
        joints = self._tracker.read_hand_state_joint(base_joints)
        return _joints_to_action(joints, self._action_names())

    def _base_joints_from_state(self, hand: str, current_state: RobotAction | None) -> np.ndarray:
        if current_state is None:
            return self._base_joints[hand]

        values: list[float] = []
        for name in self._action_names():
            if self.config.hand == "both":
                keys = (f"{hand}_{name}", name)
            else:
                keys = (name, f"{hand}_{name}")

            for key in keys:
                if key in current_state:
                    values.append(float(current_state[key]))
                    break
            else:
                return self._base_joints[hand]

        return np.asarray(values, dtype=np.float32)

    def _read_dual_hand_state_joint(self, hand: str, base_joints: np.ndarray) -> np.ndarray:
        robot_kin = None if self._viewer_hand_kinematics is None else self._viewer_hand_kinematics.get(hand)
        if robot_kin is None:
            return self._tracker.read_hand_state_joint(hand, base_joints)

        from hand_teleop.gripper_pose.gripper_pose import GripperPose

        arm_dof = robot_kin.nq
        if len(base_joints) < arm_dof + 1:
            raise ValueError(
                f"Expected at least {arm_dof + 1} base joint values for {robot_kin.urdf_path}, "
                f"got {len(base_joints)}."
            )

        arm_joints_rad = np.radians(base_joints[:arm_dof])
        gripper_val = float(base_joints[arm_dof])
        base_pose = robot_kin.fk(arm_joints_rad)
        base_gripper_pose = GripperPose.from_matrix(base_pose, open_degree=gripper_val)
        final_gripper_pose = self._tracker.read_hand_state(hand, base_gripper_pose)

        if self._tracker.safe_range:
            final_gripper_pose.clip(self._tracker.safe_range)

        new_arm_joints_rad = robot_kin.ik(arm_joints_rad.copy(), final_gripper_pose.to_matrix(), max_iters=6)
        new_arm_joints_deg = np.degrees(new_arm_joints_rad)
        return np.append(new_arm_joints_deg, final_gripper_pose.open_degree).astype(np.float32)

    def _reset_tracker_base_poses(self) -> None:
        if self._tracker is None:
            return

        if hasattr(self._tracker, "base_pose"):
            self._tracker.base_pose = None

        hands = getattr(self._tracker, "_hands", None)
        if not isinstance(hands, dict):
            return

        def reset_hands() -> None:
            for state in hands.values():
                if hasattr(state, "base_pose"):
                    state.base_pose = None

        lock = getattr(self._tracker, "_lock", None)
        if lock is None:
            reset_hands()
            return

        with lock:
            reset_hands()

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
    action = {name: float(value) for name, value in zip(action_names, joints, strict=True)}
    if "gripper.pos" in action:
        action["gripper.pos"] = _normalize_gripper_open(action["gripper.pos"])
    return action


def _normalize_gripper_open(value: float) -> float:
    return float(np.clip(abs(value), 0.0, 100.0))


class _FrameAdaptedKinematics:
    def __init__(self, robot_kin: _RobotKinematicsLike, control_to_urdf: np.ndarray):
        self._robot_kin = robot_kin
        self._control_to_urdf = control_to_urdf.astype(np.float32)

    @property
    def robot_kin(self) -> _RobotKinematicsLike:
        return self._robot_kin

    @property
    def nq(self) -> int:
        return self._robot_kin.nq

    @property
    def urdf_path(self) -> str:
        return self._robot_kin.urdf_path

    def fk(self, q: np.ndarray, frame: str | None = None) -> np.ndarray:
        return self._urdf_to_control(self._robot_kin.fk(q, frame=frame))

    def ik(self, q0: np.ndarray, target_t: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._robot_kin.ik(q0, self._control_to_urdf_pose(target_t), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot_kin, name)

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


def _unwrap_frame_adapted_kinematics(robot_kin: _RobotKinematicsLike) -> _RobotKinematicsLike:
    while isinstance(robot_kin, _FrameAdaptedKinematics):
        robot_kin = robot_kin.robot_kin
    return robot_kin
