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
import threading
import time
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.odrive import ODriveMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_gem_follower import GemFollowerConfig

if TYPE_CHECKING:
    from lerobot.processor import RobotAction, RobotObservation
else:
    RobotAction = dict[str, Any]
    RobotObservation = dict[str, Any]

logger = logging.getLogger(__name__)


class GemFollower(Robot):
    """
    Gem follower arm with mixed motors:
    - joint_1: Steadywin GIM6010 on ODrive
    - joint_2: STS3095
    - joint_3: STS3250
    - joint_4: STS3095
    - joint_5..joint_7 + gripper: STS3215
    """

    config_class = GemFollowerConfig
    name = "gem"
    _ODRIVE_AXIS = 0
    _CAPTURED_POSE_DEGREES = {
        "joint_4": -20.0,
        "joint_6": 70.0,
    }

    def __init__(self, config: GemFollowerConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self._odrive_joint = "joint_1"

        feetech_calibration = {
            k: v for k, v in self.calibration.items() if k in set(config.feetech_motor_ids.keys())
        }
        odrive_calibration = {k: v for k, v in self.calibration.items() if k == self._odrive_joint}

        self.feetech_bus = FeetechMotorsBus(
            port=config.feetech_port,
            motors={
                "joint_2": Motor(config.feetech_motor_ids["joint_2"], "sts3095", norm_mode_body),
                "joint_3": Motor(config.feetech_motor_ids["joint_3"], "sts3250", norm_mode_body),
                "joint_4": Motor(config.feetech_motor_ids["joint_4"], "sts3095", norm_mode_body),
                "joint_5": Motor(config.feetech_motor_ids["joint_5"], "sts3215", norm_mode_body),
                "joint_6": Motor(config.feetech_motor_ids["joint_6"], "sts3215", norm_mode_body),
                "joint_7": Motor(config.feetech_motor_ids["joint_7"], "sts3215", norm_mode_body),
                "gripper": Motor(config.feetech_motor_ids["gripper"], "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=feetech_calibration,
        )

        self.odrive_bus = ODriveMotorsBus(
            port=config.odrive_port,
            motors={self._odrive_joint: Motor(self._ODRIVE_AXIS, "gim6010-8", norm_mode_body)},
            calibration=odrive_calibration,
        )

        # Compatibility shim for utilities that expect `robot.bus.motors` to exist.
        self.bus = SimpleNamespace(motors={self._odrive_joint: self.odrive_bus.motors[self._odrive_joint]})
        self.bus.motors.update(self.feetech_bus.motors)

        self._odrive_lock = threading.Lock()
        self._odrive_poll_stop = threading.Event()
        self._odrive_poll_thread: threading.Thread | None = None
        self._odrive_cached_pos: float | None = None
        self._odrive_pending_target: float | None = None

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{self._odrive_joint}.pos": float} | {f"{motor}.pos": float for motor in self.feetech_bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.feetech_bus.is_connected
            and self.odrive_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.feetech_bus.connect()
        self.odrive_bus.connect()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        if calibrate:
            self.configure()
        else:
            # Keep the robot passive during explicit calibration workflows.
            self.feetech_bus.disable_torque()
            self.odrive_bus.disable_torque()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.feetech_bus.is_calibrated and self.odrive_bus.is_calibrated

    def _split_calibration(self) -> tuple[dict[str, MotorCalibration], dict[str, MotorCalibration]]:
        feetech_names = set(self.feetech_bus.motors.keys())
        feetech_calibration = {k: v for k, v in self.calibration.items() if k in feetech_names}
        odrive_calibration = {k: v for k, v in self.calibration.items() if k == self._odrive_joint}
        return feetech_calibration, odrive_calibration

    def _read_odrive_blocking(self) -> float:
        pos = float(self.odrive_bus.read("Present_Position", self._odrive_joint))
        with self._odrive_lock:
            self._odrive_cached_pos = pos
        return pos

    def _start_odrive_polling(self) -> None:
        hz = self.config.odrive_polling_hz
        if not hz or hz <= 0 or self._odrive_poll_thread is not None:
            return
        self._odrive_poll_stop.clear()
        period = 1.0 / hz
        joint = self._odrive_joint

        def _poll() -> None:
            next_tick = time.perf_counter()
            while not self._odrive_poll_stop.is_set():
                try:
                    with self._odrive_lock:
                        target = self._odrive_pending_target
                        self._odrive_pending_target = None
                    if target is not None:
                        self.odrive_bus.sync_write("Goal_Position", {joint: target})
                    self._read_odrive_blocking()
                except Exception:
                    logger.exception("%s ODrive poll error", self)
                next_tick += period
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    self._odrive_poll_stop.wait(sleep)
                else:
                    next_tick = time.perf_counter()

        self._odrive_poll_thread = threading.Thread(target=_poll, name=f"{self.id}_odrive_poll", daemon=True)
        self._odrive_poll_thread.start()

    def _stop_odrive_polling(self) -> None:
        self._odrive_poll_stop.set()
        if self._odrive_poll_thread is not None:
            self._odrive_poll_thread.join(timeout=1.0)
            self._odrive_poll_thread = None

    def _get_odrive_position(self) -> float:
        if self._odrive_poll_thread is None or self._odrive_cached_pos is None:
            return self._read_odrive_blocking()
        with self._odrive_lock:
            return self._odrive_cached_pos

    def _force_gripper_drive_mode(self, drive_mode: int = 1) -> None:
        if "gripper" not in self.calibration:
            return
        self.calibration["gripper"].drive_mode = drive_mode

    def _force_arm_ranges_full_resolution(self) -> None:
        for motor_name, m in self.feetech_bus.motors.items():
            if motor_name == "gripper" or motor_name not in self.calibration:
                continue
            max_res = self.feetech_bus.model_resolution_table[m.model] - 1
            self.calibration[motor_name].range_min = 0
            self.calibration[motor_name].range_max = max_res

    def _apply_captured_pose_biases(
        self, homing_offsets: dict[str, int], bus: FeetechMotorsBus
    ) -> dict[str, int]:
        adjusted_offsets = homing_offsets.copy()
        # The physical calibration pose is intentionally not the logical zero for these joints.
        for motor_name, captured_pose_deg in self._CAPTURED_POSE_DEGREES.items():
            if motor_name not in adjusted_offsets:
                continue
            model = bus.motors[motor_name].model
            max_res = bus.model_resolution_table[model] - 1
            ticks = int(round(captured_pose_deg * max_res / 360))
            adjusted_offsets[motor_name] -= ticks
        return adjusted_offsets

    def calibrate(self) -> None:
        self._stop_odrive_polling()
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                self._force_gripper_drive_mode(drive_mode=1)
                self._force_arm_ranges_full_resolution()
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                feetech_calibration, odrive_calibration = self._split_calibration()
                self.feetech_bus.write_calibration(feetech_calibration)
                self.odrive_bus.write_calibration(odrive_calibration)
                self._save_calibration()
                return

        logger.info(f"\nRunning calibration of {self}")
        self.odrive_bus.disable_torque()
        self.feetech_bus.disable_torque()

        for motor in self.feetech_bus.motors:
            self.feetech_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            "\nCalibration: Set Zero Position\n"
            "Position the GEM follower arm in the following configuration:\n"
            "  - Joints 1-3 hanging downward\n"
            "  - Joint 4 slightly bent (about 20 degrees below logical zero)\n"
            "  - Joint 5 centered\n"
            "  - Joint 6 pushed up to its upper limit so the hand bends upward\n"
            "  - Adjust the hand so it is parallel to the table\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )
        # Capture raw ODrive position as zero-reference for joint_1 (ignore any stale cached calibration).
        previous_odrive_calibration = self.odrive_bus.read_calibration()
        self.odrive_bus.write_calibration({}, cache=True)
        try:
            odrive_zero_pos = float(self.odrive_bus.read("Present_Position", self._odrive_joint))
        finally:
            self.odrive_bus.write_calibration(previous_odrive_calibration, cache=True)
        homing_offsets = self.feetech_bus.set_half_turn_homings()
        homing_offsets = self._apply_captured_pose_biases(homing_offsets, self.feetech_bus)
        for motor_name, offset in homing_offsets.items():
            self.feetech_bus.write("Homing_Offset", motor_name, offset)

        range_mins: dict[str, int] = {}
        range_maxes: dict[str, int] = {}
        for motor_name, m in self.feetech_bus.motors.items():
            max_res = self.feetech_bus.model_resolution_table[m.model] - 1
            range_mins[motor_name] = 0
            range_maxes[motor_name] = max_res

        input(
            "\nGripper calibration:\n"
            "Step 1/2: Move follower gripper to desired 0% position and press ENTER..."
        )
        gripper_zero = int(self.feetech_bus.read("Present_Position", "gripper", normalize=False))
        input(
            "Step 2/2: Move follower gripper to desired 100% position and press ENTER..."
        )
        gripper_max = int(self.feetech_bus.read("Present_Position", "gripper", normalize=False))
        if gripper_zero == gripper_max:
            raise ValueError("Invalid follower gripper calibration: 0% and 100% positions are identical.")
        range_mins["gripper"] = min(gripper_zero, gripper_max)
        range_maxes["gripper"] = max(gripper_zero, gripper_max)

        self.calibration = {}
        for motor_name, m in self.feetech_bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=m.id,
                drive_mode=1 if motor_name == "gripper" else 0,
                homing_offset=homing_offsets[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )

        odrive_joint = self.odrive_bus.motors[self._odrive_joint]
        self.calibration[self._odrive_joint] = MotorCalibration(
            id=odrive_joint.id,
            drive_mode=0,
            homing_offset=int(round(odrive_zero_pos)),
            range_min=-180,
            range_max=180,
        )

        feetech_calibration, odrive_calibration = self._split_calibration()
        self.feetech_bus.write_calibration(feetech_calibration)
        self.odrive_bus.write_calibration(odrive_calibration)

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.feetech_bus.torque_disabled():
            self.feetech_bus.configure_motors()
            for motor in self.feetech_bus.motors:
                self.feetech_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.feetech_bus.write("P_Coefficient", motor, 16)
                self.feetech_bus.write("I_Coefficient", motor, 0)
                self.feetech_bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.feetech_bus.write("Max_Torque_Limit", motor, 500)
                    self.feetech_bus.write("Protection_Current", motor, 250)
                    self.feetech_bus.write("Overload_Torque", motor, 25)

        self.odrive_bus.configure_motors()
        self.odrive_bus.enable_torque()
        self._read_odrive_blocking()
        self._odrive_pending_target = None
        self._start_odrive_polling()

    def setup_motors(self) -> None:
        for motor in reversed(self.feetech_bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.feetech_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.feetech_bus.motors[motor].id}")

        print("ODrive motor setup is not automated. Configure axis mapping on the ODrive board manually.")

    def _read_joint_positions(self) -> dict[str, float]:
        positions = self.feetech_bus.sync_read("Present_Position")
        positions[self._odrive_joint] = self._get_odrive_position()
        return positions

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs_dict = self._read_joint_positions()
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self._read_joint_positions()
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        if self._odrive_joint in goal_pos:
            with self._odrive_lock:
                self._odrive_pending_target = goal_pos[self._odrive_joint]

        feetech_goal_pos = {k: v for k, v in goal_pos.items() if k in self.feetech_bus.motors}
        if feetech_goal_pos:
            self.feetech_bus.sync_write("Goal_Position", feetech_goal_pos)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self._stop_odrive_polling()
        self.odrive_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.feetech_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
