#!/usr/bin/env python

import logging
import time
from functools import cached_property

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_minion_arm import BiMinionArmConfig

logger = logging.getLogger(__name__)

_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7")
_CAPTURED_POSE_DEGREES = {
    "joint_4": -20.0,
    "joint_6": 70.0,
}
_STS3215_RESOLUTION = 4096
_STS3215_MAX_RESOLUTION = 4095
_RIGHT_MIRRORED_JOINTS = ("joint_1", "joint_3", "joint_5", "joint_7")


def _wrap_homing_offset(offset: int, resolution: int = _STS3215_RESOLUTION) -> int:
    """Map an offset to the equivalent value supported by a single-turn servo."""
    half_turn = resolution // 2
    max_magnitude = half_turn - 1
    wrapped_offset = (offset + half_turn) % resolution - half_turn

    # An 11-bit sign-magnitude register cannot represent exactly half a turn.
    # Preserve the original direction and accept a one-tick error in this edge case.
    if wrapped_offset == -half_turn:
        return max_magnitude if offset >= 0 else -max_magnitude

    return wrapped_offset


def _get_gripper_calibration(
    gripper: str, zero_position: int, hundred_position: int
) -> tuple[int, int, int]:
    """Return drive mode and ascending limits for two captured gripper positions."""
    if hundred_position == zero_position:
        raise ValueError(
            f"Invalid {gripper} calibration: 0% and 100% points are both {zero_position}."
        )

    if hundred_position > zero_position:
        return 0, zero_position, hundred_position

    return 1, hundred_position, zero_position


class BiMinionArm(Teleoperator):
    """Bimanual MinionArm leader on one shared Feetech serial bus."""

    config_class = BiMinionArmConfig
    name = "bi_minion_arm"

    def __init__(self, config: BiMinionArmConfig):
        super().__init__(config)
        self.config = config
        port = config.port or config.left_arm_config.port or config.right_arm_config.port
        if not port:
            raise ValueError(
                "bi_minion_arm requires --teleop.port. "
                "--teleop.left_arm_config.port is accepted as an alias for the shared bus port."
            )
        if config.left_arm_config.port and config.right_arm_config.port and config.left_arm_config.port != config.right_arm_config.port:
            logger.warning(
                "bi_minion_arm uses one shared Feetech bus; got two ports (%s, %s). Using %s.",
                config.left_arm_config.port,
                config.right_arm_config.port,
                port,
            )

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        motors: dict[str, Motor] = {}
        for side, motor_ids in (
            ("left", config.left_arm_motor_ids),
            ("right", config.right_arm_motor_ids),
        ):
            for joint in _JOINT_NAMES:
                motors[f"{side}_{joint}"] = Motor(motor_ids[joint], "sts3215", norm_mode_body)
            motors[f"{side}_gripper"] = Motor(
                motor_ids["gripper"], "sts3215", MotorNormMode.RANGE_0_100
            )

        ids = [motor.id for motor in motors.values()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"BiMinionArm motor IDs must be unique on one bus. Got: {ids}")

        self.bus = FeetechMotorsBus(
            port=port,
            motors=motors,
            calibration=self.calibration,
        )
        self._last_action: dict[str, float] | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
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
                "Mismatch between calibration values in the motor and the calibration file or no "
                "calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def _apply_captured_pose_biases(self, homing_offsets: dict[str, int]) -> dict[str, int]:
        adjusted_offsets = homing_offsets.copy()
        for side in ("left", "right"):
            for joint, captured_pose_deg in _CAPTURED_POSE_DEGREES.items():
                motor_name = f"{side}_{joint}"
                if motor_name not in adjusted_offsets:
                    continue
                model = self.bus.motors[motor_name].model
                max_res = self.bus.model_resolution_table[model] - 1
                ticks = int(round(captured_pose_deg * max_res / 360))
                unwrapped_offset = adjusted_offsets[motor_name] - ticks
                adjusted_offsets[motor_name] = _wrap_homing_offset(unwrapped_offset, max_res + 1)
                if adjusted_offsets[motor_name] != unwrapped_offset:
                    logger.info(
                        "Wrapped %s homing offset from %d to %d",
                        motor_name,
                        unwrapped_offset,
                        adjusted_offsets[motor_name],
                    )
        return adjusted_offsets

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Writing calibration file associated with the id %s to the motors", self.id)
                self.bus.write_calibration(self.calibration)
                self._save_calibration()
                return

        logger.info("\nRunning calibration of %s", self)
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        homing_offsets = {}
        for side in ("left", "right"):
            input(
                f"\nCalibration: Set {side.upper()} Minion leader zero position\n"
                f"Position only the {side} leader arm in the following configuration:\n"
                "  - Joints 1-3 hanging downward\n"
                "  - Joint 4 slightly bent (about 20 degrees below logical zero)\n"
                "  - Joint 5 centered\n"
                "  - Joint 6 pushed up to its upper limit so the hand bends upward\n"
                "  - Adjust the hand so it is parallel to the table\n"
                "  - Gripper at desired 0% reference (usually closed)\n"
                "Press ENTER when ready..."
            )
            side_motors = [f"{side}_{joint}" for joint in _JOINT_NAMES] + [f"{side}_gripper"]
            side_offsets = self.bus.set_half_turn_homings(side_motors)
            side_offsets = self._apply_captured_pose_biases(side_offsets)
            for motor_name, offset in side_offsets.items():
                self.bus.write("Homing_Offset", motor_name, offset)
            homing_offsets.update(side_offsets)

        range_mins = dict.fromkeys(self.bus.motors, 0)
        range_maxes = dict.fromkeys(self.bus.motors, 4095)
        drive_modes = dict.fromkeys(self.bus.motors, 0)

        for gripper in ("left_gripper", "right_gripper"):
            input(f"\n{gripper} calibration: Move to desired 0% position and press ENTER...")
            gripper_zero = int(self.bus.read("Present_Position", gripper, normalize=False))
            input(f"{gripper} calibration: Move to desired 100% position and press ENTER...")
            gripper_hundred = int(self.bus.read("Present_Position", gripper, normalize=False))

            drive_mode, range_min, range_max = _get_gripper_calibration(
                gripper, gripper_zero, gripper_hundred
            )
            drive_modes[gripper] = drive_mode
            range_mins[gripper] = range_min
            range_maxes[gripper] = range_max
            if drive_mode:
                logger.info(
                    "%s encoder decreases from 0%% to 100%%; using inverted drive mode",
                    gripper,
                )

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
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

    def _apply_right_arm_mirror_convention(self, action: dict[str, float]) -> dict[str, float]:
        mirrored_action = action.copy()
        for joint in _RIGHT_MIRRORED_JOINTS:
            motor = f"right_{joint}"
            if motor in mirrored_action:
                mirrored_action[motor] = -mirrored_action[motor]
        return mirrored_action

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        try:
            if self.bus.calibration:
                action = self.bus.sync_read("Present_Position", num_retry=2)
            else:
                raw_action = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
                action = {
                    motor: (
                        (float(value) / _STS3215_MAX_RESOLUTION) * 100
                        if motor.endswith("gripper")
                        else (float(value) - (_STS3215_MAX_RESOLUTION / 2)) * 360 / _STS3215_MAX_RESOLUTION
                    )
                    for motor, value in raw_action.items()
                }
        except ConnectionError:
            if self._last_action is None:
                raise
            logger.warning("%s read action failed; reusing last action.", self)
            return dict(self._last_action)
        action = self._apply_right_arm_mirror_convention(action)
        action = {f"{motor}.pos": val for motor, val in action.items()}
        self._last_action = action
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s read action: %.1fms", self, dt_ms)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        _ = feedback
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info("%s disconnected.", self)
