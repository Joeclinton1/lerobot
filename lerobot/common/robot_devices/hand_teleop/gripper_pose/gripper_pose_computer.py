# ruff: noqa: N806 N803

import time
from typing import Literal, Optional

import cv2
import numpy as np

from lerobot.common.robot_devices.hand_teleop.gripper_pose.gripper_pose import GripperPose
from lerobot.common.robot_devices.hand_teleop.gripper_pose.kalman_filter import KalmanXYZ
from lerobot.common.robot_devices.hand_teleop.hand_keypoints.factory import (
    ModelName,
    create_estimator,
)
from lerobot.common.robot_devices.hand_teleop.hand_keypoints.types import TrackedHandKeypoints

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)

class GripperPoseComputer:
    GRIP_ANGLE_OFFSET: int = -2 # Add to raw gripper angle to ensure we can pick up small objects

    def __init__(
        self,
        device: Optional[str] = None,
        model: ModelName = "wilor",
        hand: Literal["left", "right"] = "right",
        inference_interval: int = 2,
    ):
        self.estimator = create_estimator(model, device=device)
        self.hand = hand
        self.kf = KalmanXYZ()
        self.step = 0
        self.run_every = inference_interval

        self.prev_pose = GripperPose.zero() # stores the relative vec 13 between current wilor pose and initial
        self.raw_pose: Optional[GripperPose] = None # store the raw unprocessed vector that wilor gives for visualisation purposes
        self.initial_pose = None

        self.gripper_verts = None

        self.hand_detected = False
        self.last_t = time.time()

        self.robot_axes_in_hand = np.column_stack([
            [0, 0, -1],   # new x-axis
            [-1, 0, 0],   # new y-axis
            [0, 1, 0]    # new z-axis
        ]) # This must be right handed for things to work

        self.R_hand_to_robot = self.robot_axes_in_hand.T
    
    def kf_predict(self):
        dt = time.time() - self.last_t
        self.last_t = time.time()
        self.kf.predict(dt)

    def compute_pose(self, frame: np.ndarray, focal_length: float, cam_t: np.ndarray, paused: bool) -> GripperPose:
        if paused:
            self.last_t = time.time()
            return self.prev_pose.copy()

        self.step += 1
        
        if self.hand_detected and self.step % self.run_every != 0:
            self.kf_predict()
            pose = self.prev_pose.copy()
            pose.pos = self.kf.x[:3]
            self.prev_pose = pose.copy()
            return pose

        pose = self._get_absolute_pose(frame, focal_length, cam_t)
        if pose is None:
            self.last_t = time.time()
            self.hand_detected = False
            return self.prev_pose.copy()

        self.hand_detected = True
        self.raw_pose = pose.copy() # store absolute vec13 for visualisation purposes
        pose = self._to_relative_pose(pose)
        self.kf_predict()
        self.kf.update(pose.pos)
        pose.pos = self.kf.x[:3]
        self.prev_pose = pose.copy()
        return pose

    def _get_absolute_pose(self, frame: np.ndarray, focal_length: float, cam_t: np.ndarray) -> Optional[GripperPose]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = self.estimator(frame_rgb, focal_length)
        corrected_hand = "left" if self.hand == "right" else "right"
        matching_preds = [p for p in preds if p.is_right == (corrected_hand == "right")]

        if not matching_preds:
            return None

        keypoints = matching_preds[0].keypoints
        pose = self._compute_gripper_pose(keypoints)
        self.pose_wilor = pose.copy()

        # convert from hand estimation space to robot space
        # when our hand points along the robot +x axis, which is world -z axis, it should be pointing "forwards"
        # this is strange because normally pointing along the robot z axis is forward
        # to achieve this our change of basis does a similarity transform, so the axis are relabelled
        # but forward is along the hand frame z axis (robot x) instead of robot z
        pose.change_basis(self.robot_axes_in_hand, cam_t)
        return pose

    def _to_relative_pose(self, abs_pose: GripperPose) -> GripperPose:
        pose = abs_pose.copy()

        if self.initial_pose is None:
            self.initial_pose = pose.copy()

        # we aim to find the pose which takes us from the initial pose to the abs_pose
        # this is equivalent to applying an inverse transform to our pose with the initial pose
        pose.inverse_transform_pose(self.initial_pose.rot, self.initial_pose.pos)
        return pose

    def _compute_gripper_pose(self, kp: TrackedHandKeypoints) -> GripperPose:
        # we assume the hand estimation is in screen space
        # x axis points from screen left to screen right
        # y axis points from screen bottom to screen top
        # z axis points out of the screen (or well into the screen in the reflection? It's confusing)
        x_raw = normalize(kp.middle_base - kp.index_base)
        y_axis = normalize(kp.index_base - kp.thumb_mcp)
        z_axis = normalize(np.cross(x_raw, y_axis))
        x_axis = normalize(np.cross(y_axis, z_axis))
        R_hand = np.column_stack([x_axis, y_axis, z_axis]) 

        origin = 0.5 * (kp.index_base + kp.middle_base)

        tip = 0.5 * (kp.index_tip + kp.middle_tip)
        vec1 = tip - origin

        plane_n = x_axis
        thumb_root = kp.thumb_mcp - np.dot(kp.thumb_mcp - origin, plane_n) * plane_n
        thumb_tip_proj = kp.thumb_tip - np.dot(kp.thumb_tip - origin, plane_n) * plane_n
        vec2 = normalize(thumb_tip_proj - thumb_root) * np.linalg.norm(vec1)
        vec3 = normalize(thumb_tip_proj - origin) * np.linalg.norm(vec1)

        closed_angle = np.degrees(
            np.arccos(np.clip(np.dot(normalize(vec3), normalize(vec2)), -1, 1))
        )
        closed_angle *= np.sign(np.dot(np.cross(vec2, vec3), plane_n))

        angle_rad = np.arccos(np.clip(np.dot(normalize(vec2), normalize(vec1)), -1.0, 1.0))
        grip_angle = np.degrees(angle_rad) * np.sign(np.dot(np.cross(vec2, vec1), plane_n))
        grip_angle -= closed_angle
        grip_angle += self.GRIP_ANGLE_OFFSET

        gripper_pose = GripperPose(
            origin,
            R_hand,
            grip_angle,
            [origin.copy(), tip,thumb_root, thumb_tip_proj]
        )
        
        return gripper_pose