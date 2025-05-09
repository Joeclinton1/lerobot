import time
from math import radians as rad

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from lerobot.common.robot_devices.vision_teleop.hand_pose import GRIP_ID, POS_SL, ROT_SL, rotmat_lh_to_rpy_zyx
from lerobot.common.robot_devices.vision_teleop.hand_tracking import HandTracker

DEBUG = True

def main():
    tracker = HandTracker(cam_idx=1, hand="right")
    _ema_fps = None

    # Define follower rest pose: [x y z | R (row-major) | gripper]
    follower_pos = [0.2, 0, 0.1]
    yaw, pitch, roll = rad(0), rad(45), rad(-90)
    follower_grip = 5

    follower_rot = R.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    follower_vec13 = np.concatenate([
        follower_pos,
        follower_rot.flatten(),
        [follower_grip]
    ]).astype(np.float32)

    alpha = 0.1
    while tracker.cap.isOpened():
        t0 = time.time()
        try:
            vec13 = tracker.read_hand_state(follower_vec13)
        except RuntimeError:
            break

        fps = 1.0 / max(time.time() - t0, 1e-6)
        _ema_fps = (
            fps if _ema_fps is None else (1 - alpha) * _ema_fps + alpha * fps
        )

        if DEBUG:
            if vec13 is not None:
                pos = vec13[POS_SL]
                rotmat = vec13[ROT_SL].reshape(3, 3)
                eul = np.degrees(rotmat_lh_to_rpy_zyx(rotmat))
                grip = vec13[GRIP_ID]
                print(f"Pos: {pos.round(2)}, Euler: {eul.round(1)}, Grip: {grip:.1f} | ", end="")
            print(f"FPS: {_ema_fps:.1f}")

        if cv2.waitKey(1) in (27, ord("q")):
            break

    tracker.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()