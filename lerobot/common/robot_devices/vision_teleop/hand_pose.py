# ruff: noqa: N806 N803

from typing import Literal, Optional

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

from lerobot.common.robot_devices.vision_teleop.kalman import KalmanXYZ

# ee_vec = [ x y z | R(3×3) row–major | gripper ]
EE_LEN = 13
POS_SL = slice(0, 3)
ROT_SL = slice(3, 12)
GRIP_ID = 12

def rotmat_lh_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_euler("ZYX")[::-1]

class HandPoseComputer:
    FOCAL_RATIO: float = 0.7
    CAM_ORIGIN = np.array([0, 0.24, 0.6])
    GRIP_ANGLE_OFFSET: int = -2 # Subtracted from the raw gripper angle to ensure we can pick up small objects

    def __init__(
        self,
        device: Optional[str] = None,
        hand: Literal["left", "right"] = "right",
        inference_interval: int = 2
    ):
        self.pipe = WiLorHandPose3dEstimationPipeline(
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float16,
            verbose=False,
        )
        self.hand = hand
        self.kf = KalmanXYZ()
        self.step = 0
        self.run_every = inference_interval

        self.prev_vec = self.zero_vec13() # stores the relative vec 13 between current wilor pose and initial
        self.raw_vec = None # store the raw unprocessed vector that wilor gives for visualisation purposes
        self.initial_pos = None
        self.initial_R = None
        self.initial_follower_vec = None

        self.last_info = None
        self.last_focal2 = 12500
        self.cam_pos = None
        self.z_scale = None

        self.hand_detected = False

        self.R_wilor_to_robot = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ])

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.zeros_like(v)

    @staticmethod
    def zero_vec13() -> np.ndarray:
        vec13 = np.zeros(EE_LEN, dtype=np.float32)
        vec13[ROT_SL] = np.eye(3, dtype=np.float32).reshape(-1)
        return vec13

    def compute_pose(self, frame: np.ndarray, paused: bool, dt: float) -> np.ndarray:
        if paused:
            return self.prev_vec.copy()

        self.step += 1
        self.kf.predict(dt)
        if self.hand_detected and self.step % self.run_every != 0:
            vec13 = self.prev_vec.copy()
            vec13[POS_SL] = self.kf.x[:3]
            self.prev_vec = vec13.copy()
            return vec13

        vec13 = self._get_absolute_pose(frame, dt)
        if vec13 is None:
            self.hand_detected = False
            return self.prev_vec.copy()

        self.hand_detected = True
        self.raw_vec = vec13.copy() # store absolute vec13 for visualisation purposes
        vec13 = self._to_relative_pose(vec13)
        self.kf.update(vec13[POS_SL])
        vec13[POS_SL] = self.kf.x[:3]
        self.prev_vec = vec13.copy()
        return vec13

    def compose_absolute_pose(self, rel_pose: np.ndarray, follower_vec: np.ndarray) -> np.ndarray:
        if self.initial_follower_vec is None:
            self.initial_follower_vec = follower_vec.copy()

        result = rel_pose.copy()
        result[POS_SL] += follower_vec[POS_SL]

        R_base = follower_vec[ROT_SL].reshape(3, 3)
        R_rel = result[ROT_SL].reshape(3, 3)
        result[ROT_SL] = (R_rel @ R_base).reshape(-1)
        return result

    def _get_absolute_pose(self, frame: np.ndarray, dt: float) ->  Optional[np.ndarray]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = self.pipe.predict(frame_rgb)
        corrected_hand = "left" if self.hand == "right" else "right"
        matching_preds = [p for p in preds if p["is_right"] == (corrected_hand == "right")]

        if not matching_preds:
            return None

        p = matching_preds[0]
        focal = self.FOCAL_RATIO * frame.shape[1]
        self.last_focal2 = p["wilor_preds"]["scaled_focal_length"]
        self.z_scale = focal / self.last_focal2
        self.cam_pos = self.CAM_ORIGIN / np.array([1, 1, -self.z_scale])

        kps3 = p["wilor_preds"]["pred_keypoints_3d"][0] + p["wilor_preds"]["pred_cam_t_full"][0]
        kps3 *= (1, -1, 1)
        kps3 += self.cam_pos

        info, vec13 = self._compute_gripper_pose(kps3, self.z_scale)
        self.last_info = info
        return vec13

    def _to_relative_pose(self, abs_pose: np.ndarray) -> np.ndarray:
        pose = abs_pose.copy()

        if self.initial_pos is None:
            self.initial_pos = pose[POS_SL].copy()
            self.initial_R = pose[ROT_SL].reshape(3, 3).copy()

        pos_rel = pose[POS_SL] - self.initial_pos
        R_rel = self.initial_R.T @ pose[ROT_SL].reshape(3, 3)

        pose[POS_SL] = pos_rel
        pose[ROT_SL] = R_rel.reshape(-1)

        return pose

    def _compute_gripper_pose(self, kps3: np.ndarray, z_scale: float) -> tuple[dict, np.ndarray]:
        thumb_mcp = kps3[2]
        thumb_tip = kps3[4]
        index_base = kps3[5]
        index_tip = kps3[8]
        middle_base = kps3[9]
        middle_tip = kps3[12]

        x_raw = self._normalize(middle_base - index_base)
        y_axis = self._normalize(thumb_mcp - index_base)
        z_axis = self._normalize(np.cross(x_raw, y_axis))
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        R_wilor = np.row_stack([x_axis, y_axis, z_axis])

        R_robot = self.R_wilor_to_robot @ R_wilor @ self.R_wilor_to_robot.T
        origin = 0.5 * (index_base + middle_base)

        base = origin
        tip = 0.5 * (index_tip + middle_tip)
        vec1 = tip - base

        plane_n = x_axis
        thumb_root = thumb_mcp - np.dot(thumb_mcp - origin, plane_n) * plane_n
        thumb_tip_proj = thumb_tip - np.dot(thumb_tip - origin, plane_n) * plane_n
        vec2 = self._normalize(thumb_tip_proj - thumb_root) * np.linalg.norm(vec1)

        vec3 = self._normalize(thumb_tip_proj - origin) * np.linalg.norm(vec1)
        closed_angle = np.degrees(
            np.arccos(np.clip(np.dot(self._normalize(vec3), self._normalize(vec2)), -1, 1))
        )
        closed_angle *= np.sign(np.dot(np.cross(vec2, vec3), plane_n))

        angle_rad = np.arccos(np.clip(np.dot(self._normalize(vec1), self._normalize(vec2)), -1.0, 1.0))
        grip_angle = np.degrees(angle_rad) * np.sign(np.dot(np.cross(vec2, vec1), plane_n))
        grip_angle -= closed_angle
        grip_angle += self.GRIP_ANGLE_OFFSET

        origin_t = origin * np.array([-1, 1, -z_scale])
        origin_t = origin_t[[2, 0, 1]]
        vec13 = np.concatenate([
            origin_t,
            R_robot.flatten(),
            [grip_angle]
        ]).astype(np.float32)

        return {
            "tip": tip,
            "thumb_root": thumb_root,
            "thumb_proj": vec2,
        }, vec13