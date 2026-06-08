from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("none")
@RobotConfig.register_subclass("no_follower")
@dataclass
class NoneRobotConfig(RobotConfig):
    """No-op robot used when teleoperation drives only the viewer."""

    pass
