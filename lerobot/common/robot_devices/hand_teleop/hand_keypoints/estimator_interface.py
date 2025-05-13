from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from lerobot.common.robot_devices.hand_teleop.hand_keypoints.types import HandKeypointsPred


class HandPoseEstimator(ABC):
    @abstractmethod
    def __init__(self, device: Optional[str] = None):
        """
        device: The compute device (e.g., 'cuda', 'cpu')
        """
        pass

    @abstractmethod
    def __call__(self, image: np.ndarray, focal_length: float) -> list[HandKeypointsPred]:
        """
        Run pose estimation on the given BGR image.

        Returns:
            A list of HandPosePrediction objects with world-space keypoints
        """
        pass