#!/usr/bin/env python

from dataclasses import dataclass
from typing import Literal

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("hand_teleop")
@dataclass
class HandTeleopConfig(TeleoperatorConfig):
    """Webcam hand tracking teleoperator backed by hand-teleop."""

    cam_idx: int = 0
    hand: Literal["left", "right", "both"] = "right"
    model: Literal["wilor", "mediapipe", "apriltag"] = "wilor"
    device: str | None = None
    show_viz: bool = False
    start_paused: bool = False
    fps: int = 30
    urdf_path: str = "so100"
    frame_name: str = "gripper_link"
    focal_ratio: float = 0.7
    use_scroll: bool = False
    left_base_joint: tuple[float, float, float, float, float, float] | None = None
    right_base_joint: tuple[float, float, float, float, float, float] | None = None
