# ruff: noqa: N806 N803

import time
from math import radians as rad
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from pynput import keyboard
from scipy.spatial.transform import Rotation
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

# ee_vec = [ x y z | R(3×3) row–major | gripper ]
EE_LEN = 13          # 3 + 9 + 1
POS_SL  = slice(0, 3)
ROT_SL  = slice(3, 12)
GRIP_ID = 12

class HandPoseTracker:
    # ── configuration ──────────────────────────────────────────────
    DEBUG: bool = False
    FOCAL_RATIO: float = 0.7
    CAM_ORIGIN = np.array([0, 0.24, 0.6])

    def __init__(
        self,
        cam_idx: int = 0,
        device: Optional[str] = None,
        show_viz: bool = True,
        hand: Literal["left", "right"] = "right"
    ):
        self.cap = cv2.VideoCapture(cam_idx)
        self.pipe = WiLorHandPose3dEstimationPipeline(
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float16,
            verbose=False,
        )
        self.show_viz = show_viz
        self.hand = hand
        self._ema_fps = None
        self.initial_pos: Optional[np.ndarray] = None
        self.initial_R:   Optional[np.ndarray] = None
        self.initial_follower_vec: Optional[np.ndarray] = None
        
        self.prev_vec: np.ndarray = self.zero_vec13()
        self.cam_pos = None
        self.tracking_paused = True # Start in a paused state. Press space to start.
        listener = keyboard.Listener(on_press=self.onpress, on_release=self.onrelease)
        listener.start()

        self.R_wilor_to_robot = np.array([
            [0, 0, 1],  # new X = old Z
            [-1, 0, 0],  # new Y = old X
            [0, -1, 0],  # new Z = old Y
        ]) # Wilor uses a standard opengl convention, but our robot arm kinematics expects this instead.


    # ───────────────────────── math helpers ────────────────────────
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.zeros_like(v)
    
    @staticmethod
    def rotmat_lh_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
        """
        Returns [roll, pitch, yaw] using intrinsic ZYX order for a left-handed matrix.
        """
        return Rotation.from_matrix(R).as_euler("ZYX")[::-1]
    
    @staticmethod
    def zero_vec13() -> np.ndarray:
        vec13 = np.zeros(EE_LEN, dtype=np.float32)
        vec13[ROT_SL] = np.eye(3, dtype=np.float32).reshape(-1)
        return vec13

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

    def _project_points_3d_to_2d(self, pts3d: np.ndarray, image_shape, focal: float = 600):
        h, w = image_shape
        fx = fy = focal
        cx, cy = w / 2, h / 2
        pts3d = pts3d - self.cam_pos
        Z = pts3d[:, 2] + 1e-6
        xs = fx * pts3d[:, 0] / Z + cx
        ys = -fy * pts3d[:, 1] / Z + cy
        return np.column_stack((xs, ys)).astype(np.int32)

    def _compute_gripper_pose(
        self, kps3: np.ndarray, z_scale: float, is_right: bool
    ) -> dict:
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
    
        R_robot = self.R_wilor_to_robot @ R_wilor @ self.R_wilor_to_robot.T  # rexpresses hand orientation in robot world frame instead instead of wilor world frame
        axes = self.R_wilor_to_robot  @ R_wilor
        # R *= np.array([[1, -1, -1]] if is_right else [[-1, 1, -1]]) # hack to fix the axes orientation
        origin = 0.5 * (index_base + middle_base)

        base = origin
        tip = 0.5 * (index_tip + middle_tip)
        vec1 = tip - base

        plane_n = x_axis
        thumb_root = thumb_mcp - np.dot(thumb_mcp - origin, plane_n) * plane_n
        thumb_tip_proj = thumb_tip - np.dot(thumb_tip - origin, plane_n) * plane_n
        vec2 = self._normalize(thumb_tip_proj - thumb_root) * np.linalg.norm(vec1)

        # Closed angle (reference grip direction)
        vec3 = self._normalize(thumb_tip_proj - origin) * np.linalg.norm(vec1)
        closed_angle = np.degrees(
            np.arccos(np.clip(np.dot(self._normalize(vec3), self._normalize(vec2)), -1, 1))
        )
        closed_angle *= np.sign(np.dot(np.cross(vec2, vec3), plane_n))

        angle_rad = np.arccos(np.clip(np.dot(self._normalize(vec1), self._normalize(vec2)), -1.0, 1.0))
        grip_angle = np.degrees(angle_rad) * np.sign(np.dot(np.cross(vec2, vec1), plane_n))
        grip_angle -= closed_angle

        origin_t = origin * np.array([-1, 1, -z_scale])
        origin_t = origin_t[[2, 0, 1]]
        vec13 = np.concatenate([origin_t,
                            R_robot.flatten(),        # 9 numbers, row‑major
                            [grip_angle]]).astype(np.float32)

        return {
            "origin": origin,
            "axes": axes,
            "tip": tip,
            "thumb_root": thumb_root,
            "thumb_proj": vec2,
            "angle": grip_angle,
            "vec13": vec13,
        }

    # --‑‑‑‑ drawing helpers (kept mostly unchanged) ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    def _draw_3d_line(self, im, start, end, color, focal, arrow=True):
        pts = self._project_points_3d_to_2d(
            np.vstack([start, end]), im.shape[:2], focal
        )
        if arrow:
            cv2.arrowedLine(im, tuple(pts[0]), tuple(pts[1]), color, 2, tipLength=0.2)
        else:
            cv2.line(im, tuple(pts[0]), tuple(pts[1]), color, 2)

    def _draw_gripper_axes(self, im, origin, axes, focal, length=0.03):
        # X (red), Y (green), Z (blue) in BGR
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(3):  # 0: X, 1: Y, 2: Z
            self._draw_3d_line(im, origin, origin + axes[i] * length, colors[i], focal)
        return im

    def _draw_gripper_vectors(self, im, base, tip, thumb_root, thumb_proj, focal):
        self._draw_3d_line(im, base, tip, (0, 255, 255), focal, True)
        self._draw_3d_line(
            im, thumb_root, thumb_root + thumb_proj, (255, 255, 0), focal, True
        )
        self._draw_3d_line(im, thumb_root, base, (180, 180, 180), focal, False)
        return im
    
    def _draw_gripper_info_text(self, im, base, vec13, focal):
        pos = self._project_points_3d_to_2d(base[None], im.shape[:2], focal)[0]

        # position
        x, y, z = vec13[POS_SL]
        cv2.putText(im, f"x:{x:.2f}, y:{y:.2f}, z:{z:.2f}",
                    tuple(pos + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # orientation (convert only for display)
        R = vec13[ROT_SL].reshape(3,3)
         
        roll, pitch, yaw = np.degrees(self.rotmat_lh_to_rpy_zyx(R))
        cv2.putText(im, f"u:{roll:.0f}, v:{pitch:.0f}, w:{yaw:.0f}",
                    tuple(pos + np.array([10, 10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # gripper
        cv2.putText(im, f"{vec13[GRIP_ID]:.1f} deg",
                    tuple(pos + np.array([10, 30])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return im

    # ───────────────────────── public API ───────────────────────────
    def read_hand_state(self, follower_vec13) -> Optional[np.ndarray]:
        # Read and flip the camera frame
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        frame = cv2.flip(frame, 1)
        hand = "left" if self.hand == "right" else "left"  # mirror correction

        R_rel = None
        # Set or copy follower pose baseline
        if self.initial_follower_vec is None:
            initial_follower_vec = follower_vec13.copy()
            self.initial_follower_vec = initial_follower_vec
        else:
            initial_follower_vec = self.initial_follower_vec.copy()

        # Handle paused tracking
        if self.tracking_paused:
            result = self.prev_vec.copy()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = self.pipe.predict(frame_rgb)
            matching_preds = [p for p in preds if p["is_right"] == (hand == "right")]

            if not matching_preds:
                result = self.prev_vec.copy()
            else:
                p = matching_preds[0]
                focal = self.FOCAL_RATIO * frame.shape[1]
                z_scale = focal / p["wilor_preds"]["scaled_focal_length"]
                self.cam_pos = self.CAM_ORIGIN / np.array([1, 1, -z_scale])

                kps3 = (
                    p["wilor_preds"]["pred_keypoints_3d"][0]
                    + p["wilor_preds"]["pred_cam_t_full"][0]
                )
                kps3 *= (1, -1, 1)
                kps3 += self.cam_pos

                info = self._compute_gripper_pose(kps3, z_scale, p["is_right"])
                vec13 = info["vec13"]

                # Offset to make output relative to initial position
                if self.initial_pos is None:
                    self.initial_pos = vec13[POS_SL].copy()
                    self.initial_R = vec13[ROT_SL].reshape(3, 3).copy()

                # relative pose
                vec13[POS_SL] -= self.initial_pos
                R_now = vec13[ROT_SL].reshape(3, 3).copy()
                R_rel = self.initial_R.T.copy() @ R_now 
                vec13[ROT_SL] = R_rel.reshape(-1)

                self.prev_vec = vec13.copy()
                result = vec13

        # Compose final absolute pose
        result[POS_SL] += initial_follower_vec[POS_SL]

        R_base = initial_follower_vec[ROT_SL].reshape(3, 3)
        R_final = result[ROT_SL].reshape(3, 3) @ R_base
        result[ROT_SL] = R_final.reshape(-1)

        # NOTE: gripper remains unmodified (taken from hand)

        # Draw visualisation using final composed pose
        if self.show_viz and not self.tracking_paused and matching_preds:
            focal2 = p["wilor_preds"]["scaled_focal_length"]
            frame = self._draw_gripper_axes(frame, info["origin"], info["axes"], focal2)
            frame = self._draw_gripper_vectors(
                frame, info["origin"], info["tip"], info["thumb_root"], info["thumb_proj"], focal2
            )
            # if(R_rel is not None):
            #     print(R_rel.reshape(-1))
            #     result[ROT_SL] = R_rel.reshape(-1)
            #     frame = self._draw_gripper_info_text(frame, info["origin"], result, focal2)
            frame = self._draw_gripper_info_text(frame, info["origin"], result, focal2)

        # Show the frame
        if self.show_viz:
            cv2.imshow("WiLor Hand Pose 3D", frame)
            cv2.waitKey(1)

        return result

    def loop(self):
        from scipy.spatial.transform import Rotation as R  # noqa: N817

        # Define follower rest pose in 13D format: [x y z | R (row-major) | gripper]
        follower_pos = [0.2, 0, 0.1]

        # Desired orientation: yaw→pitch→roll = [0°, 45°, -90°]
        yaw, pitch, roll = rad(0), rad(45), rad(-90)
        follower_grip = 5

        # ZYX for an intrinsic rotation around yaw → pitch → roll
        follower_rot = R.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()

        # Flatten the matrix and build the full 13D vector
        follower_vec13 = np.concatenate([
            follower_pos,
            follower_rot.flatten(),
            [follower_grip]
        ]).astype(np.float32)

        alpha = 0.1
        while self.cap.isOpened():
            t0 = time.time()
            try:
                vec13 = self.read_hand_state(follower_vec13)
            except RuntimeError:
                break

            fps = 1.0 / max(time.time() - t0, 1e-6)
            self._ema_fps = (
                fps if self._ema_fps is None else (1 - alpha) * self._ema_fps + alpha * fps
            )

            if self.DEBUG:
                if vec13 is not None:
                    pos = vec13[POS_SL]
                    rotmat = vec13[ROT_SL].reshape(3,3)
                    eul = np.degrees(self.rotmat_lh_to_rpy_zyx(rotmat))
                    grip = vec13[GRIP_ID]
                    print(f"Pos: {pos.round(2)}, Euler: {eul.round(1)}, Grip: {grip:.1f} | ", end="")
                print(f"FPS: {self._ema_fps:.1f}")

            if cv2.waitKey(1) in (27, ord("q")):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HandPoseTracker(cam_idx=1, hand="right").loop()