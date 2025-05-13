from typing import Optional

import numpy as np
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

from lerobot.common.robot_devices.hand_teleop.hand_keypoints.estimator_interface import (
    HandPoseEstimator,
)
from lerobot.common.robot_devices.hand_teleop.hand_keypoints.types import (
    HandKeypointsPred,
    TrackedHandKeypoints,
)


class WiLorEstimator(HandPoseEstimator):
    def __init__(self, device: Optional[str] = None):
        import torch
        self.pipe = WiLorHandPose3dEstimationPipeline(
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float16,
            verbose=False,
        )

    def __call__(self, image: np.ndarray, focal_len: float) -> list[HandKeypointsPred]:
        raw_preds = self.pipe.predict(image)
        preds = []
        for p in raw_preds:
            kp = p["wilor_preds"]["pred_keypoints_3d"][0] + p["wilor_preds"]["pred_cam_t_full"][0]
            kp *= (1, -1, 1) # Wilor coordinates need to be flipped in the y axis, to ensure a right handed coordinate system
            
            # the z coordinate predicted by wilor is arbitary
            # The hand's z-scale is correct, but it's arbitarily shifted in the z-axis so we need to correct for this
            focal_scale = focal_len / p["wilor_preds"]["scaled_focal_length"]
            base = (kp[5]+ kp[9])/2
            rescaled_origin = base * np.array([1,1,focal_scale])
            kp = kp - base + rescaled_origin

            preds.append(HandKeypointsPred(
                is_right=p["is_right"],
                keypoints=TrackedHandKeypoints(
                    thumb_mcp=kp[2],
                    thumb_tip=kp[4],
                    index_base=kp[5],
                    index_tip=kp[8],
                    middle_base=kp[9],
                    middle_tip=kp[12],
                )
            ))
        return preds