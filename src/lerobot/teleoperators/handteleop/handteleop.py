import time
import logging

from .config_handteleop import HandTeleopConfig
from ..teleoperator import Teleoperator
from hand_teleop.tracking.tracker import HandTracker
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)

class HandTeleop(Teleoperator):
    config_class = HandTeleopConfig
    name = "handteleop"

    def __init__(self, config: HandTeleopConfig):
        super().__init__(config)
        self.hand_tracker = HandTracker(
            model=config.model,
            cam_idx=config.cam_idx,
            show_viz=config.show_viz,
            hand=config.hand,
            urdf_path=config.urdf_path,
            use_scroll=config.use_scroll,
            safe_range=config.safe_range,
            debug_mode=config.debug_mode,
            kf_q=config.kf_q,
            kf_r=config.kf_r,
        )
        self._connected = True
        self._calibrated = True

    @property
    def action_features(self):
        return {"hand_position": list}

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        self._calibrated = True

    def configure(self) -> None:
        pass

    def get_action(self, current_state: Optional[dict]=None) -> dict:
        start = time.perf_counter()
        if current_state is None:
            raise ValueError("HandTeleop.get_action requires current_state argument.")
        
        current_state_np = np.array(list(current_state.values()), dtype=np.float32)
        hand_state_as_joint = self.hand_tracker.read_hand_state_joint(current_state_np)
        action = {k:hand_state_as_joint[i] for i, k in enumerate(current_state)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict) -> None:
        raise NotImplementedError


    def disconnect(self) -> None:
        self._connected = False
