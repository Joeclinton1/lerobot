#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import TeleoperatorConfig


class PhoneOS(Enum):
    ANDROID = "android"
    IOS = "ios"


class PhoneArm(Enum):
    LEFT = "left"
    RIGHT = "right"


@TeleoperatorConfig.register_subclass("phone")
@dataclass
class PhoneConfig(TeleoperatorConfig):
    phone_os: PhoneOS = PhoneOS.IOS
    arm: PhoneArm = PhoneArm.LEFT
    # Android WebXR reports raw phone axes as +Z top edge, +X left edge,
    # and +Y screen normal. After teleop's R/U/B -> F/L/U transform,
    # calibration yields physical top=-x, left=-y, screen=+z. GEM
    # viewer/control coordinates are +Y up, +X along the shoulder,
    # and the left-arm front is -Z.
    target_x_axis: str = "y"
    target_y_axis: str = "z"
    target_z_axis: str = "x"
    target_x_sign: float = 1.0
    target_y_sign: float = 1.0
    target_z_sign: float = 1.0
    position_scale: float = 0.5
    orientation_scale: float = 1.0
    # IK soft-task weights. A large position weight keeps the solved pose from lagging behind the
    # target (the soft-task steady-state offset scales with 1/position_weight); orientation is kept
    # at a fixed ratio below it so the wrist still tracks. Both must stay well above the kinematics
    # posture weight so redundancy resolution never pulls the end-effector off target.
    position_weight: float = 200.0
    orientation_weight: float = 4.0
    camera_offset = np.array(
        [0.0, -0.02, 0.04]
    )  # iPhone 14 Pro camera is 2cm off center and 4cm above center
