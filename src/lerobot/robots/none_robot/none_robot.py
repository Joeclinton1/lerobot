from typing import Any

from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from .config_none_robot import NoneRobotConfig


class NoneRobot(Robot):
    """No-op robot that accepts actions without talking to hardware."""

    config_class = NoneRobotConfig
    name = "none"

    def __init__(self, config: NoneRobotConfig):
        super().__init__(config)
        self._is_connected = False

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {}

    @property
    def action_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        _ = calibrate
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def get_observation(self) -> RobotObservation:
        return {}

    def send_action(self, action: RobotAction) -> RobotAction:
        return dict(action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        _ = feedback

    def disconnect(self) -> None:
        self._is_connected = False
