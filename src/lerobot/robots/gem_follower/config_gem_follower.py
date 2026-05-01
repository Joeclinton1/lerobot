#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class GemFollowerConfigBase:
    """Configuration for the Gem follower arm (mixed ODrive + Feetech)."""

    # USB/serial port used by the Feetech chain (joint_2..joint_7 + gripper)
    feetech_port: str

    # ODrive selector. Use "auto"/"any" to pick the first found board, or set an ODrive serial number.
    odrive_port: str = "auto"

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits action deltas for safety.
    max_relative_target: float | dict[str, float] | None = None

    # Cameras attached to the follower robot.
    # Expected camera names: "head" and "wrist".
    # Example: --robot.cameras='{head: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}'
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Keep using degree-based interfaces for joints.
    use_degrees: bool = True

    # Feetech IDs for the 7 downstream joints.
    # joint_2: sts3095, joint_3: sts3250, joint_4: sts3095, joint_5..gripper: sts3215
    feetech_motor_ids: dict[str, int] = field(
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


@RobotConfig.register_subclass("gem")
@RobotConfig.register_subclass("gem_follower")
@dataclass
class GemFollowerConfig(RobotConfig, GemFollowerConfigBase):
    pass
