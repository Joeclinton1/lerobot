import argparse
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from lerobot.common.robot_devices.hand_teleop.gripper_pose.gripper_pose import GripperPose
from lerobot.common.robot_devices.hand_teleop.tracker import HandTracker


def main(quiet=False):
    tracker = HandTracker(cam_idx=1, hand="right")
    _ema_fps = None

    # Define follower rest pose
    follower_pos = np.array([0.2, 0, 0.1])
    follower_rot = R.from_euler("ZYX", [0, 45, -90], degrees=True).as_matrix() # in robot world frame
    follower_vec13 = GripperPose(follower_pos, follower_rot, open_degree=5)

    alpha = 0.1
    while tracker.cap.isOpened():
        t0 = time.time()
        try:
            pose = tracker.read_hand_state(follower_vec13)
        except RuntimeError:
            break

        fps = 1.0 / max(time.time() - t0, 1e-6)
        _ema_fps = (
            fps if _ema_fps is None else (1 - alpha) * _ema_fps + alpha * fps
        )

        if not quiet:
            if pose is not None:
                pos = pose.pos
                eul = pose.rot_euler
                grip = pose.open_degree
                print(f"Pos: {pos.round(2)}, Euler: {eul.round(1)}, Grip: {grip:.1f} | ", end="")
            print(f"FPS: {_ema_fps:.1f}")

        if cv2.waitKey(1) in (27, ord("q")):
            break

    tracker.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()
    main(quiet=args.quiet)