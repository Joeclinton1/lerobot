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

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_minion_arm import MinionArmConfig

logger = logging.getLogger(__name__)


class MinionArm(Teleoperator):
    """
    MinionArm leader teleoperator: 7 DOF + gripper, all Feetech STS3215.
    """

    config_class = MinionArmConfig
    name = "minion_arm"
    _CAPTURED_POSE_DEGREES = {
        "joint_4": -20.0,
        "joint_6": 70.0,
    }

    def __init__(self, config: MinionArmConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "sts3215", norm_mode_body),
                "joint_2": Motor(2, "sts3215", norm_mode_body),
                "joint_3": Motor(3, "sts3215", norm_mode_body),
                "joint_4": Motor(4, "sts3215", norm_mode_body),
                "joint_5": Motor(5, "sts3215", norm_mode_body),
                "joint_6": Motor(6, "sts3215", norm_mode_body),
                "joint_7": Motor(7, "sts3215", norm_mode_body),
                "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def _force_gripper_drive_mode(self, drive_mode: int = 0) -> None:
        if "gripper" not in self.calibration:
            return
        self.calibration["gripper"].drive_mode = drive_mode

    def _apply_captured_pose_biases(self, homing_offsets: dict[str, int]) -> dict[str, int]:
        adjusted_offsets = homing_offsets.copy()
        # The physical calibration pose is intentionally not the logical zero for these joints.
        for motor_name, captured_pose_deg in self._CAPTURED_POSE_DEGREES.items():
            if motor_name not in adjusted_offsets:
                continue
            model = self.bus.motors[motor_name].model
            max_res = self.bus.model_resolution_table[model] - 1
            ticks = int(round(captured_pose_deg * max_res / 360))
            adjusted_offsets[motor_name] -= ticks
        return adjusted_offsets

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                self._force_gripper_drive_mode(drive_mode=0)
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                self._save_calibration()
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            "\nCalibration: Set Zero Position\n"
            "Position the Minion leader arm in the following configuration:\n"
            "  - Joints 1-3 hanging downward\n"
            "  - Joint 4 slightly bent (about 20 degrees below logical zero)\n"
            "  - Joint 5 centered\n"
            "  - Joint 6 pushed up to its upper limit so the hand bends upward\n"
            "  - Adjust the hand so it is parallel to the table\n"
            "  - Gripper at desired 0% reference (usually closed)\n"
            "Press ENTER when ready..."
        )
        homing_offsets = self.bus.set_half_turn_homings()
        homing_offsets = self._apply_captured_pose_biases(homing_offsets)
        for motor_name, offset in homing_offsets.items():
            self.bus.write("Homing_Offset", motor_name, offset)
        # OpenArm-like calibration flow: zero capture + fixed limits.
        # For Feetech 12-bit position registers, use full raw range.
        range_mins = {motor: 0 for motor in self.bus.motors}
        range_maxes = {motor: 4095 for motor in self.bus.motors}
        # On Minion, joint_5 is the full-turn wrist-rotation joint.
        range_mins["joint_5"] = 0
        range_maxes["joint_5"] = 4095

        input(
            "\nGripper calibration:\n"
            "Step 1/2: Move gripper to desired 0% position and press ENTER..."
        )
        gripper_zero = int(self.bus.read("Present_Position", "gripper", normalize=False))

        input(
            "Step 2/2: Move gripper to desired 100% position and press ENTER..."
        )
        gripper_max = int(self.bus.read("Present_Position", "gripper", normalize=False))

        if gripper_max <= gripper_zero:
            raise ValueError(
                "Invalid gripper calibration points for Minion leader with drive_mode=0: "
                f"100% point ({gripper_max}) must be greater than 0% point ({gripper_zero})."
            )

        range_mins["gripper"] = gripper_zero
        range_maxes["gripper"] = gripper_max

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        _ = feedback
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
