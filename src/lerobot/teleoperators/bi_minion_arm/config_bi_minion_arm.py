#!/usr/bin/env python

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class BiMinionArmPortConfig:
    port: str = ""


@TeleoperatorConfig.register_subclass("bi_minionarm")
@TeleoperatorConfig.register_subclass("bi_minion_arm")
@dataclass
class BiMinionArmConfig(TeleoperatorConfig):
    """Configuration for two MinionArm leaders on one shared Feetech bus."""

    port: str = ""
    left_arm_config: BiMinionArmPortConfig = field(default_factory=BiMinionArmPortConfig)
    right_arm_config: BiMinionArmPortConfig = field(default_factory=BiMinionArmPortConfig)
    use_degrees: bool = True
    left_arm_motor_ids: dict[str, int] = field(
        default_factory=lambda: {
            "joint_1": 1,
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
            "joint_1": 9,
            "joint_2": 10,
            "joint_3": 11,
            "joint_4": 12,
            "joint_5": 13,
            "joint_6": 14,
            "joint_7": 15,
            "gripper": 16,
        }
    )
