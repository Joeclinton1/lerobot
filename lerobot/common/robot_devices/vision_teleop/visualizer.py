import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.common.robot_devices.vision_teleop.hand_pose import GRIP_ID, POS_SL, ROT_SL


class HandPoseVisualizer:
    def __init__(self, r_wilor_to_robot: np.ndarray, cam_origin: np.ndarray):
        self.R_wilor_to_robot = r_wilor_to_robot
        self.cam_origin = cam_origin
        self.cam_pos = None

    def _project_points_3d_to_2d(self, pts3d: np.ndarray, image_shape, focal: float):
        h, w = image_shape
        fx = fy = focal
        cx, cy = w / 2, h / 2
        pts3d = pts3d - self.cam_pos
        Z = pts3d[:, 2] + 1e-6  # noqa: N806
        xs = fx * pts3d[:, 0] / Z + cx
        ys = -fy * pts3d[:, 1] / Z + cy
        return np.column_stack((xs, ys)).astype(np.int32)

    def _draw_3d_line(self, im, start, end, color, focal, arrow=True):
        pts = self._project_points_3d_to_2d(np.vstack([start, end]), im.shape[:2], focal)
        if arrow:
            cv2.arrowedLine(im, tuple(pts[0]), tuple(pts[1]), color, 2, tipLength=0.2)
        else:
            cv2.line(im, tuple(pts[0]), tuple(pts[1]), color, 2)

    def draw_gripper_axes(self, im, origin, axes, focal, length=0.03):
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X, Y, Z
        for i in range(3):
            self._draw_3d_line(im, origin, origin + axes[i] * length, colors[i], focal)
        return im

    def draw_gripper_vectors(self, im, base, tip, thumb_root, thumb_proj, focal):
        self._draw_3d_line(im, base, tip, (0, 255, 255), focal, True)
        self._draw_3d_line(im, thumb_root, thumb_root + thumb_proj, (255, 255, 0), focal, True)
        self._draw_3d_line(im, thumb_root, base, (180, 180, 180), focal, False)
        return im

    def draw_gripper_info_text(self, im, base, vec13, focal):
        pos = self._project_points_3d_to_2d(base[None], im.shape[:2], focal)[0]
        x, y, z = vec13[POS_SL]
        R = vec13[ROT_SL].reshape(3, 3)  # noqa: N806
        roll, pitch, yaw = np.degrees(Rotation.from_matrix(R).as_euler("ZYX")[::-1])

        cv2.putText(im, f"x:{x:.2f}, y:{y:.2f}, z:{z:.2f}", tuple(pos + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(im, f"u:{roll:.0f}, v:{pitch:.0f}, w:{yaw:.0f}", tuple(pos + [10, 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(im, f"{vec13[GRIP_ID]:.1f} deg", tuple(pos + [10, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return im
    
    def draw_all(self, frame, origin_pos, axes, vec13, info: dict, focal, cam_pos):
        self.cam_pos = cam_pos
        frame = self.draw_gripper_axes(frame, origin_pos, axes, focal)
        frame = self.draw_gripper_vectors(frame, origin_pos,
                                        info["tip"], info["thumb_root"], info["thumb_proj"], focal)
        frame = self.draw_gripper_info_text(frame, origin_pos, vec13, focal)
        return frame
