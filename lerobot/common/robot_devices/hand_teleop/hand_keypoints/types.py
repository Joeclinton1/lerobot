from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedHandKeypoints:
    thumb_mcp: np.ndarray
    thumb_tip: np.ndarray
    index_base: np.ndarray
    index_tip: np.ndarray
    middle_base: np.ndarray
    middle_tip: np.ndarray


@dataclass
class HandKeypointsPred:
    is_right: bool
    keypoints: TrackedHandKeypoints
