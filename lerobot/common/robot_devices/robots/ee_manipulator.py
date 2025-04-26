"""
EE‑space utilities (Euler‑angle edition)
========================================
This module contains two closely‑related utilities:

* ``EEManipulatorDecorator`` – wraps a *ManipulatorRobot* to expose its
  control and observation interface in **end‑effector space** using Euler
  angles instead of quaternions:

      [x, y, z, roll, pitch, yaw, gripper]

  where the angles are intrinsic **XYZ** (roll‑pitch‑yaw) in *radians*.

* ``render_debug_scene`` – a lightweight Matplotlib visualiser that plots
  the current EE observation (red) against a commanded EE action (green).

Both pieces rely on the same helper conversions between joint‑space and
EE‑space as well as Euler <‑‑> rotation‑matrix utilities.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lerobot.common.robot_devices.robots.kinematics import RobotKinematics

# -----------------------------------------------------------------------------
# Global switch for console logging + live visualisation
# -----------------------------------------------------------------------------
DEBUG = False

# -----------------------------------------------------------------------------
# EE‑space ⇄ SE(3) helpers
# -----------------------------------------------------------------------------

def pose_to_vec(ee_pose: np.ndarray, gripper: float) -> np.ndarray:
    """Convert a 4×4 homogeneous pose into our 7‑D EE vector."""
    pos = ee_pose[:3, 3]
    euler = Rotation.from_matrix(ee_pose[:3, :3]).as_euler("xyz")
    return np.concatenate([pos, euler, [gripper]]).astype(np.float32)


def vec_to_pose(vec: np.ndarray) -> tuple[np.ndarray, float]:
    """Inverse of :func:`pose_to_vec`. Returns (4×4 pose, gripper)."""
    if vec.shape[-1] != 7:
        raise ValueError("EE vector must have 7 elements [x,y,z,r,p,y,gripper]")
    pos, euler, gripper = vec[:3], vec[3:6], vec[6]
    ee_pose = np.eye(4)
    ee_pose[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
    ee_pose[:3, 3] = pos
    return ee_pose, gripper


# -----------------------------------------------------------------------------
# EE‑space decorator
# -----------------------------------------------------------------------------

class EEManipulatorDecorator:
    """Decorator adding EE‑space control/observation to a *ManipulatorRobot*."""

    def __init__(self, robot, ee_mode: str = "abs"):
        assert ee_mode in {"abs", "delta"}, "ee_mode must be 'abs' or 'delta'"
        self.robot = robot
        self.kinematics = RobotKinematics(robot.robot_type)
        self.ee_mode = ee_mode
        self._leader_ee_ref: np.ndarray | None = None

        if DEBUG:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")

    # ------------------------------------------------------------------
    # Attribute proxying – expose the wrapped robot as‑is
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        return getattr(self.robot, name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_ee(self, joint_state: torch.Tensor) -> np.ndarray:
        ee_pose = self.kinematics.fk_gripper(joint_state.numpy())
        return pose_to_vec(ee_pose, joint_state[-1].item())

    def _delta_pose(self, ee_pose: np.ndarray) -> np.ndarray:
        if self._leader_ee_ref is None:
            self._leader_ee_ref = ee_pose.copy()
            return ee_pose

        delta = ee_pose - self._leader_ee_ref
        desired = self._leader_ee_ref.copy()
        desired[:3, 3] += delta[:3, 3]  # position‑only delta
        self._leader_ee_ref = ee_pose
        return desired

    # ------------------------------------------------------------------
    # Public API mirroring the wrapped robot
    # ------------------------------------------------------------------
    def capture_observation(self):
        obs = self.robot.capture_observation()
        obs["observation.state"] = torch.from_numpy(self._to_ee(obs["observation.state"]))
        return obs

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # 1) EE → joint command
        ee_pose, gripper = vec_to_pose(action.numpy())
        current = torch.from_numpy(self.robot.follower_arms["main"].read("Present_Position"))
        desired = ee_pose if self.ee_mode == "abs" else self._delta_pose(ee_pose)
        cmd = self.kinematics.ik(current.numpy(), desired, position_only=False)
        cmd_t = torch.from_numpy(np.concatenate([cmd[:-1], [gripper]]).astype(np.float32))

        # 2) send & grab what *really* went out
        sent_joints = self.robot.send_action(cmd_t)

        # 3) FK → EE and return
        sent_pose = self.kinematics.fk_gripper(sent_joints.numpy())
        ee_vec = pose_to_vec(sent_pose, sent_joints[-1].item())
        return torch.from_numpy(ee_vec.astype(np.float32))

    def teleop_step(self, record_data: bool = False):
        def _fmt(x):
            return " ".join(f"{v:.3f}" for v in x)

        obs, action = self.robot.teleop_step(record_data=record_data)

        if DEBUG:
            print("joint obs:", _fmt(obs["observation.state"]),
                  "| joint act:", _fmt(action["action"]))

        obs["observation.state"] = torch.from_numpy(self._to_ee(obs["observation.state"]))
        action["action"] = torch.from_numpy(self._to_ee(action["action"]))

        if DEBUG:
            print("ee obs:", _fmt(obs["observation.state"]),
                  "| ee act:", _fmt(action["action"]), "\n")

            render_debug_scene(obs["observation.state"], action["action"], ax=self.ax)

        return obs, action

    @staticmethod
    def get_ee_space_features() -> dict[str, dict]:
        names = [
            "ee_x", "ee_y", "ee_z",
            "ee_roll", "ee_pitch", "ee_yaw",
            "ee_gripper"
        ]
        # pull in any existing camera/image features
        features = {
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": names
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": names
            },
        }
        return features


# -----------------------------------------------------------------------------
# Debug visualisation helpers
# -----------------------------------------------------------------------------

def _vec_to_transform(vec: np.ndarray) -> np.ndarray:
    """Convert our 7‑D EE vector into a 4×4 transform."""
    pos = vec[:3]
    euler = vec[3:6]
    R = Rotation.from_euler("xyz", euler).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _draw_origin(ax, length: float = 0.1):
    ax.quiver(0, 0, 0, length, 0, 0, color="r", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, length, 0, color="g", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, length, color="b", arrow_length_ratio=0.1)


def _draw_cube(ax, T: np.ndarray, color: str, size: float = 0.1):
    half = size / 2.0
    verts = np.array([
        [-half, -half, -half],
        [ half, -half, -half],
        [ half,  half, -half],
        [-half,  half, -half],
        [-half, -half,  half],
        [ half, -half,  half],
        [ half,  half,  half],
        [-half,  half,  half],
    ])
    verts_h = np.hstack((verts, np.ones((8, 1))))  # homogeneous
    verts_w = (T @ verts_h.T).T[:, :3]

    faces = [
        [verts_w[j] for j in [0, 1, 2, 3]],
        [verts_w[j] for j in [4, 5, 6, 7]],
        [verts_w[j] for j in [0, 1, 5, 4]],
        [verts_w[j] for j in [2, 3, 7, 6]],
        [verts_w[j] for j in [1, 2, 6, 5]],
        [verts_w[j] for j in [4, 7, 3, 0]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=0.5, edgecolors="k"))

    # Draw an arrow along local +X (front face)
    front = (T @ np.array([half, 0, 0, 1]))[:3]
    center = T[:3, 3]
    ax.quiver(*center, *(front - center), color="k", arrow_length_ratio=0.3)


def render_debug_scene(obs_vec: np.ndarray, act_vec: np.ndarray, ax=None):
    """Visualise EE observation vs action in 3‑D (red vs. green)."""
    T_obs = _vec_to_transform(obs_vec)
    T_act = _vec_to_transform(act_vec)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.cla()
    _draw_origin(ax, length=0.1)
    _draw_cube(ax, T_obs, color="red", size=0.1)
    _draw_cube(ax, T_act, color="green", size=0.1)

    # Axis limits: X & Z non‑negative, Y symmetric
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Debug Scene: EE Obs (Red) vs Act (Green)")

    plt.pause(0.001)