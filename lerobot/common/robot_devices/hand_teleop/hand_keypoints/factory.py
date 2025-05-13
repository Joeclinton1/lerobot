from typing import Literal, Type

from lerobot.common.robot_devices.hand_teleop.hand_keypoints.estimator_interface import (
    HandPoseEstimator,
)
from lerobot.common.robot_devices.hand_teleop.hand_keypoints.estimators.wilor_estimator import (
    WiLorEstimator,
)

# from .estimators.mediapipe_estimator import MediaPipeEstimator

ModelName = Literal["wilor"]  # static for type checkers

_REGISTRY: dict[str, Type[HandPoseEstimator]] = {
    "wilor": WiLorEstimator,
    # "mediapipe": MediaPipeEstimator,
}

def create_estimator(
    name: ModelName = "wilor",
    device=None,
    **kwargs
) -> HandPoseEstimator:

    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown estimator '{name}'. Available: {list(_REGISTRY)}")
    return cls(device=device, **kwargs)
