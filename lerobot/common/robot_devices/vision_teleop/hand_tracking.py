# ruff: noqa: N806 N803

import time
from typing import Literal, Optional

import cv2
import numpy as np
from pynput import keyboard

from lerobot.common.robot_devices.vision_teleop.hand_pose import POS_SL, ROT_SL, HandPoseComputer
from lerobot.common.robot_devices.vision_teleop.visualizer import HandPoseVisualizer


class HandTracker:
    def __init__(
        self,
        cam_idx: int = 0,
        device: Optional[str] = None,
        hand: Literal["left", "right"] = "right",
        show_viz: bool = True,
    ):
        self.cap = cv2.VideoCapture(cam_idx)
        self.hand = hand
        self.show_viz = show_viz
        self.tracking_paused = True
        self.last_t = time.time()
        self.step = 0
        self._ema_fps = None

        self.pose_computer = HandPoseComputer(device=device, hand=hand, inference_interval=2)
        self.visualizer = HandPoseVisualizer(self.pose_computer.R_wilor_to_robot, self.pose_computer.CAM_ORIGIN)
        listener = keyboard.Listener(on_press=self.onpress, on_release=self.onrelease)
        listener.start()

    # ───────────────────────── math helpers ────────────────────────
  

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
        self.initial_pos = None
        self.initial_follower_vec = None

    # ───────────────────────── public API ───────────────────────────
    def read_hand_state(self, follower_vec13) -> Optional[np.ndarray]:
        # Read and mirror camera frame
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        frame = cv2.flip(frame, 1)

        # Update frame timing
        self.step += 1
        dt = time.time() - self.last_t
        self.last_t = time.time()

        # Pose estimation (relative)
        rel_pose = self.pose_computer.compute_pose(frame, paused=self.tracking_paused, dt=dt)

        # Combine with follower baseline
        result = self.pose_computer.compose_absolute_pose(rel_pose, follower_vec13)

        # Optional debug visualisation
        if self.show_viz and not self.tracking_paused and self.pose_computer.last_info is not None:
            focal2 = self.pose_computer.last_focal2
            origin_scale = np.array([-1, 1, -1 / self.pose_computer.z_scale])
            origin_pos = self.pose_computer.prev_vec[POS_SL][[1, 2, 0]] * origin_scale
            axes = self.pose_computer.prev_vec[ROT_SL].reshape(3, 3) @ self.pose_computer.R_wilor_to_robot

            frame = self.visualizer.draw_all(
                frame,
                origin_pos,
                axes,
                result,
                self.pose_computer.last_info,
                focal2,
                self.pose_computer.cam_pos
            )

        if self.show_viz:
            cv2.imshow("WiLor Hand Pose 3D", frame)
            cv2.waitKey(1)
        
        return result