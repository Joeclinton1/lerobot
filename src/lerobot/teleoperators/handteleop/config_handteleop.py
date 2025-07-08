from dataclasses import dataclass
from typing import Literal, Optional, Dict
from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("handteleop")
@dataclass(kw_only=True)
class HandTeleopConfig(TeleoperatorConfig):
    model: Literal["wilor", "mediapipe", "apriltag"] = "wilor"
    cam_idx: int = 0
    show_viz: bool = True
    hand: Literal["right", "left"] = "right"
    urdf_path: str = "so100"
    use_scroll: bool = False
    kf_q: float = 5e-4
    kf_r: float = 2e-2
    safe_range: Optional[Dict[str, tuple]] = None
    debug_mode: bool = False
