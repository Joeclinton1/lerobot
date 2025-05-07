import cv2
import numpy as np
import torch
import time
from typing import Literal, Optional
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
from pynput import keyboard
from math import radians as rad
from math import degrees as deg

class HandPoseTracker:
    # ── configuration ──────────────────────────────────────────────
    DEBUG: bool = True
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
        self.initial_vec: Optional[np.ndarray] = None
        self.initial_follower_vec: Optional[np.ndarray] = None
        self.prev_vec: np.ndarray = np.zeros(7)
        self.cam_pos = None
        self.tracking_paused = False
        listener = keyboard.Listener(
            on_press=lambda key: self.stop_tracking() if key == keyboard.Key.space else None,
            on_release=lambda key: self.restart_tracking() if key == keyboard.Key.space else None,
        )
        listener.start()


    # ───────────────────────── math helpers ────────────────────────
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.zeros_like(v)

    @staticmethod
    def _mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
        sy = np.hypot(R[0, 0], R[1, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        if sy < 1e-6:
            return np.arctan2(-R[1, 2], R[1, 1]), pitch, 0.0
        return (
            np.arctan2(R[2, 1], R[2, 2]),
            pitch,
            np.arctan2(R[1, 0], R[0, 0]),
        )
    
    @staticmethod
    def _mat_to_rpy_xyz(R: np.ndarray) -> tuple[float, float, float]:
        sp = R[0, 2]
        pitch = -np.arcsin(sp)
        cp = np.sqrt(1 - sp ** 2)
        if cp < 1e-6:
            roll = np.arctan2(-R[1, 0], R[1, 1])
            yaw = 0.0
        else:
            roll = np.arctan2(R[1, 2], R[2, 2])
            yaw = np.arctan2(R[0, 1], R[0, 0])
        return roll, pitch, yaw
    
    def stop_tracking(self):
        self.tracking_paused = True

    def restart_tracking(self):
        self.tracking_paused = False
        self.initial_vec = None
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
        z_axis = self._normalize(np.cross(y_axis, x_raw))
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        R = np.column_stack([x_axis, y_axis, z_axis])
        R = R[:, [2, 0, 1]]
        R *= np.array([[1, -1, -1]] if is_right else [[-1, 1, -1]]) # hack to fix the axes orientation

        roll, pitch, yaw = self._mat_to_rpy_xyz(R)
        roll, pitch, yaw = -pitch, roll, yaw # Another hack to fix orientations
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
        closed_angle *= np.sign(np.dot(np.cross(vec3, vec2), plane_n))

        angle_rad = np.arccos(np.clip(np.dot(self._normalize(vec1), self._normalize(vec2)), -1.0, 1.0))
        grip_angle = np.degrees(angle_rad) * np.sign(np.dot(np.cross(vec1, vec2), plane_n))
        grip_angle -= closed_angle

        origin_t = origin * np.array([-1, 1, -z_scale])
        origin_t = origin_t[[2, 0, 1]]
        vec7 = np.array([*origin_t, roll, pitch, yaw, grip_angle], dtype=np.float32)

        return dict(
            origin=origin,
            axes=R,
            tip=tip,
            thumb_root=thumb_root,
            thumb_proj=vec2,
            angle=grip_angle,
            vec7=vec7,
        )

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
        for i, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            self._draw_3d_line(
                im, origin, origin + axes[:, i] * length, col, focal, arrow=True
            )
        return im

    def _draw_gripper_vectors(self, im, base, tip, thumb_root, thumb_proj, focal):
        self._draw_3d_line(im, base, tip, (0, 255, 255), focal, True)
        self._draw_3d_line(
            im, thumb_root, thumb_root + thumb_proj, (255, 255, 0), focal, True
        )
        self._draw_3d_line(im, thumb_root, base, (180, 180, 180), focal, False)
        return im
    
    def _draw_gripper_info_text(self, im, base, vec7, focal):
        pos = self._project_points_3d_to_2d(
            base[None], im.shape[:2], focal
        )[0]
        # Gripper Position
        cv2.putText(
            im,
            f"x:{vec7[0]:.2f}, y:{vec7[1]:.2f}, z:{vec7[2]:.2f}",
            tuple(pos + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Gripper Orientation
        cv2.putText(
            im,
            f"u:{deg(vec7[3]):.0f}, v:{deg(vec7[4]):.0f}, w:{deg(vec7[5]):.0f}",
            tuple(pos + np.array([10, 10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Gripper Angle
        cv2.putText(im, f"{vec7[6]:.1f} deg", tuple(pos + np.array([10, 30])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return im

        

    # ───────────────────────── public API ───────────────────────────
    def read_hand_state(self, follower_vec7) -> Optional[np.ndarray]:
        # Read and flip the camera frame
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        frame = cv2.flip(frame, 1)
        hand = "left" if self.hand == "right" else "left"  # mirror correction
 
        if self.initial_follower_vec is None:
            initial_follower_vec = follower_vec7.copy()
            self.initial_follower_vec = initial_follower_vec
        else:
            # snapshot initial_follower_vec to avoid race condition with restart_tracking
            initial_follower_vec = self.initial_follower_vec.copy()

        # Handle paused tracking
        if self.tracking_paused:
            result = self.prev_vec.copy()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = self.pipe.predict(frame_rgb)
            matching_preds = [p for p in preds if p["is_right"] == (hand == "right")]

            # Handle non-detection of hand
            if not matching_preds:
                result = np.zeros(7)
            else:
                p = matching_preds[0]
                # Transform 3D keypoints into camera space
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
                vec7 = info["vec7"]

                # Offset to make output relative to initial position
                if self.initial_vec is None:
                    self.initial_vec = vec7.copy()

                vec7[:3] = vec7[:3] - self.initial_vec[:3]
                vec7[3:6] = vec7[3:6] - self.initial_vec[3:6] # Maybe euler angles won't work this nicely?
                self.prev_vec = vec7.copy()
                result = vec7

                # Add gripper visualization
                if self.show_viz:
                    focal2 = p["wilor_preds"]["scaled_focal_length"]
                    frame = self._draw_gripper_axes(
                        frame, 
                        info["origin"], 
                        info["axes"], 
                        focal2
                    )
                    frame = self._draw_gripper_vectors(
                        frame,
                        info["origin"],
                        info["tip"],
                        info["thumb_root"],
                        info["thumb_proj"],
                        focal2
                    )
                    frame = self._draw_gripper_info_text(
                        frame,
                        info["origin"],
                        vec7,
                        focal2
                    )

        # Display result frame
        if self.show_viz:
            cv2.imshow("WiLor Hand Pose 3D", frame)
            cv2.waitKey(1)

        result[:6] += initial_follower_vec[:6] # we apply our position and rotation to the initial follower vec
        return result

    def loop(self):
        
        follower_vec7 = np.array([20,0,1,rad(-90), rad(45), rad(0), 5]) # Approximately the follower rest position
        alpha = 0.1
        while self.cap.isOpened():
            t0 = time.time()
            try:
                vec7 = self.read_hand_state(follower_vec7)
            except RuntimeError:
                break

            fps = 1.0 / max(time.time() - t0, 1e-6)
            self._ema_fps = (
                fps if self._ema_fps is None else (1 - alpha) * self._ema_fps + alpha * fps
            )
            if self.DEBUG:
                if vec7 is not None:
                    print(f"Vec: {vec7.round(3)} | ", end="")
                print(f"FPS: {self._ema_fps:.1f}")

            if cv2.waitKey(1) in (27, ord("q")):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HandPoseTracker(cam_idx=1, hand="right").loop()