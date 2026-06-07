#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class EloNeckConfig:
    """ELO neck servos on the shared Feetech bus."""

    motor_ids: dict[str, int] = field(
        default_factory=lambda: {
            "pan": 50,
            "nod_left": 51,
            "nod_right": 52,
        }
    )
    centers: dict[str, int] = field(
        default_factory=lambda: {
            "pan": 2048,
            "nod_left": 2045,
            "nod_right": 3048,
        }
    )
    pan_limit: int = 1024
    tilt_limit: int = 512
    deadzone_degrees: float = 2.0


@RobotConfig.register_subclass("elo")
@RobotConfig.register_subclass("bi_gem")
@RobotConfig.register_subclass("bi_gem_follower")
@dataclass
class BiGemFollowerConfig(RobotConfig):
    """ELO / bimanual Gem follower: shared Feetech bus, two SteadyWin ODrive buses, optional neck."""

    feetech_port: str
    left_odrive_port: str = "auto"
    right_odrive_port: str = "auto"
    left_odrive_axis: int = 0
    right_odrive_axis: int = 0
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True
    neck: EloNeckConfig | None = field(default_factory=EloNeckConfig)
    left_arm_motor_ids: dict[str, int] = field(
        default_factory=lambda: {
            "joint_2": 2,
            "joint_3": 3,
            "joint_4": 4,
            "joint_5": 5,
            "joint_6": 6,
            "joint_7": 7,
            "gripper": 8,
        }
    )
    right_arm_motor_ids: dict[str, int] = field(
        default_factory=lambda: {
            "joint_2": 10,
            "joint_3": 11,
            "joint_4": 12,
            "joint_5": 13,
            "joint_6": 14,
            "joint_7": 15,
            "gripper": 16,
        }
    )
