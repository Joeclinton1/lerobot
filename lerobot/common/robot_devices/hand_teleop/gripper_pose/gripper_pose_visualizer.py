import cv2
import numpy as np

from lerobot.common.robot_devices.hand_teleop.gripper_pose.gripper_pose import GripperPose
from scipy.spatial.transform import Rotation as R  # noqa: N817

class GripperPoseVisualizer:
    def __init__(self, robot_axes_in_screen: np.ndarray, focal_ratio: float, cam_t:np.ndarray):
        self.robot_axes_in_screen = robot_axes_in_screen
        self.R_screen_to_robot = self.robot_axes_in_screen.T
        self.focal_ratio = focal_ratio
        self.cam_t = cam_t

        self.focal_length = 600
        

    def _project_points_3d_to_2d(self, pts3d: np.ndarray, image_shape):
        h, w = image_shape
        fx = fy = self.focal_length
        cx, cy = w / 2, h / 2
        Z = pts3d[:, 2] + 1e-6  # noqa: N806
        xs = fx * pts3d[:, 0] / Z + cx
        ys = -fy * pts3d[:, 1] / Z + cy # we need to flip it because the screen y is flipped compared to the hand tracking frame
        return np.column_stack((xs, ys)).astype(np.int32)

    def _draw_3d_line(self, im, start, end, color, arrow=True):
        pts = self._project_points_3d_to_2d(np.vstack([start, end]), im.shape[:2])
        if arrow:
            cv2.arrowedLine(im, tuple(pts[0]), tuple(pts[1]), color, 2, tipLength=0.2)
        else:
            cv2.line(im, tuple(pts[0]), tuple(pts[1]), color, 2)

    def draw_gripper_axes(
        self,
        im: np.ndarray,
        origin: np.ndarray,             # shape: (3,)
        axes: np.ndarray,               # shape: (3, 3)
        length: float = 0.03
    ) -> np.ndarray:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X, Y, Z
        for i in range(3):
            self._draw_3d_line(im, origin, origin + axes[:, i] * length, colors[i])
        return im

    def draw_gripper_vectors(
        self,
        im: np.ndarray,
        top_tail: np.ndarray,          # shape: (3,)
        top_head: np.ndarray,          # shape: (3,)
        bottom_tail: np.ndarray,       # shape: (3,)
        bottom_head: np.ndarray        # shape: (3,)
    ) -> np.ndarray:
        self._draw_3d_line(im, top_tail, top_head, (0, 255, 255), True)
        self._draw_3d_line(im, bottom_tail, bottom_head, (255, 255, 0), True)
        self._draw_3d_line(im, bottom_tail, top_tail, (180, 180, 180), False)
        return im

    def draw_gripper_info_text(
        self,
        im: np.ndarray,
        anchor_pos: np.ndarray,               # shape: (3,)
        pose: GripperPose
    ) -> np.ndarray:
        pos_2d = self._project_points_3d_to_2d(anchor_pos[None], im.shape[:2])[0]

        x,y,z = pose.pos
        roll, pitch, yaw = pose.rot_euler
        open_degree = pose.open_degree

        cv2.putText(im, f"x:{x:.2f}, y:{y:.2f}, z:{z:.2f}", tuple(pos_2d + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(im, f"u:{roll:.0f}, v:{pitch:.0f}, w:{yaw:.0f}", tuple(pos_2d + [10, 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(im, f"{open_degree:.1f} deg", tuple(pos_2d + [10, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return im
    
    def draw_all(
        self,
        frame: np.ndarray,
        pose_raw: GripperPose,
        pose_final: GripperPose,
        pose_rel: GripperPose,
        pose_base: GripperPose
    ) -> np.ndarray:
        
        # for visualizing the points position we just want to revert the basis back to the screen
        pose_screen = pose_raw.copy()
        pose_screen.revert_basis(self.robot_axes_in_screen, self.cam_t)

        # Pose_final is the hand rotation in the robot local frame but with forward vector starting at the hand world frame forward
        # if we visualise this then when the hand points forwards, the axes match the hand axes rather than robot axes
        # inorder to see what the robot axes look like we need to apply the robot axes transformation to the final pose
        pose_applied = pose_final.copy()
        pose_applied.transform_pose(self.robot_axes_in_screen,np.array([0,0,0]))

        # pose_applied2 = pose_rel.copy()
        # pose_applied2.transform_pose(self.robot_axes_in_screen,np.array([0,0,0]))

      
        # pose_applied3 = pose_raw.copy()
        # pose_applied3.transform_pose(self.robot_axes_in_screen,np.array([0,0,0]))

        # rot = R.from_euler("ZYX", [0, -45, 90], degrees=True).as_matrix() # in robot world frame
        # print(pose_rel.rot)
        # print(rot)
        # pose_applied4 = pose_base.copy()
        # pose_applied4.transform_pose(rot, np.array([0,0,0]))
        # pose_applied4_s = pose_applied4.copy()
        # pose_applied4_s.transform_pose(self.robot_axes_in_screen,np.array([0,0,0]))

        # draw all visualisations to frame
        self.focal_length = self.focal_ratio * frame.shape[1]
        frame = self.draw_gripper_axes(frame, pose_screen.pos, pose_applied.rot)
        frame = self.draw_gripper_vectors(frame, *pose_screen._keypoints)
        frame = self.draw_gripper_info_text(frame, pose_screen.pos, pose_final)
        return frame
