# ruff: noqa: N806 N803

from typing import Literal, Optional

import cv2
import numpy as np
from pynput import keyboard

from lerobot.common.robot_devices.hand_teleop.gripper_pose.gripper_pose import GripperPose
from lerobot.common.robot_devices.hand_teleop.gripper_pose.gripper_pose_computer import GripperPoseComputer
from lerobot.common.robot_devices.hand_teleop.gripper_pose.gripper_pose_visualizer import (
    GripperPoseVisualizer,
)
from scipy.spatial.transform import Rotation as R  # noqa: N817

DEFAULT_CAM_T = np.array([0, -0.24, 0.6], dtype=np.float32) # in the hand coordinate frame

class HandTracker:
    def __init__(
        self,
        cam_idx: int = 0,
        device: Optional[str] = None,
        hand: Literal["left", "right"] = "right",
        show_viz: bool = True,
        focal_ratio: float = 0.7,
        cam_t: np.ndarray = DEFAULT_CAM_T
    ):
        self.focal_ratio = focal_ratio
        self.hand = hand
        self.show_viz = show_viz
        self.cam_t = cam_t

        self.cap = cv2.VideoCapture(cam_idx)
        self.tracking_paused = True
        self._ema_fps = None
        self.base_pose: Optional[GripperPose] = None
       
        self.pose_computer = GripperPoseComputer(device=device, hand=hand, inference_interval=1)
        self.pose_visualizer = GripperPoseVisualizer(self.pose_computer.robot_axes_in_hand, self.focal_ratio, self.cam_t)

        listener = keyboard.Listener(on_press=self.onpress, on_release=self.onrelease)
        listener.start()
    

    # ───────────────────────── keyboard controls ────────────────────────

    def onpress(self, key):
        if key == keyboard.Key.space:
            self.stop_tracking()
        elif key == keyboard.KeyCode.from_char('p'):
            if self.tracking_paused:
                self.restart_tracking()
            else:
                self.stop_tracking()
    
    def onrelease(self, key):
        if key == keyboard.Key.space:
            self.restart_tracking()
        
    def stop_tracking(self):
        self.tracking_paused = True

    def restart_tracking(self):
        self.tracking_paused = False
        self.pose_computer.initial_pose = None
        self.base_pose = None

    # ───────────────────────── public API ───────────────────────────
    def read_hand_state(self, base_pose: GripperPose) -> GripperPose:
        # Read and mirror camera frame
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        frame = cv2.flip(frame, 1)

        # Pose estimation (relative)
        pose_rel = self.pose_computer.compute_pose(
            frame,
            self.focal_ratio * frame.shape[1],
            self.cam_t,
            self.tracking_paused
        )

        # Combine with follower baseline
        if self.base_pose is None:
            self.base_pose = base_pose.copy()

        # debug_pos = np.array([0.2, 0, 0.1])
        # debug_rot = R.from_euler("ZYX", [0, 90, -90], degrees=True).as_matrix() # in robot world frame
        # debug_pose = GripperPose(debug_pos, debug_rot, open_degree=5)

        pose_final = self.base_pose.copy()
        pose_final.transform_pose(pose_rel.rot, pose_rel.pos)
        # pose_final.transform_pose(debug_pose.rot, debug_pose.pos)
        pose_final.open_degree = pose_rel.open_degree

        # Optional debug visualisation
        if self.show_viz and not self.tracking_paused and self.pose_computer.raw_pose is not None:
            frame = self.pose_visualizer.draw_all(
                frame,
                self.pose_computer.raw_pose,
                pose_final,
                pose_rel,
                self.base_pose
            )

        if self.show_viz:
            cv2.imshow("WiLor Hand Pose 3D", frame)
            cv2.waitKey(1)
        
        return pose_final