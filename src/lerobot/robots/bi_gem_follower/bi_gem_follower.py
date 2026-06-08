#!/usr/bin/env python

import logging
from functools import cached_property
from types import SimpleNamespace

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.odrive import ODriveMotorsBus
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_bi_gem_follower import BiGemFollowerConfig

logger = logging.getLogger(__name__)

_ARM_JOINT_MODELS = {
    "joint_2": "sts3095",
    "joint_3": "sts3250",
    "joint_4": "sts3095",
    "joint_5": "sts3215",
    "joint_6": "sts3215",
    "joint_7": "sts3215",
}
_CAPTURED_POSE_DEGREES = {
    "joint_4": -20.0,
    "joint_6": 70.0,
}


def _deg_to_steps(degrees: float, limit: int) -> int:
    return int(max(-limit, min(limit, degrees / 90.0 * limit)))


class BiGemFollower(Robot):
    """ELO / bimanual Gem follower with one Feetech bus and two ODrive buses."""

    config_class = BiGemFollowerConfig
    name = "elo"

    def __init__(self, config: BiGemFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        feetech_motors: dict[str, Motor] = {}
        for side, motor_ids in (
            ("left", config.left_arm_motor_ids),
            ("right", config.right_arm_motor_ids),
        ):
            for joint, model in _ARM_JOINT_MODELS.items():
                feetech_motors[f"{side}_{joint}"] = Motor(motor_ids[joint], model, norm_mode_body)
            feetech_motors[f"{side}_gripper"] = Motor(
                motor_ids["gripper"], "sts3215", MotorNormMode.RANGE_0_100
            )

        if config.neck is not None:
            feetech_motors |= {
                "neck_pan": Motor(config.neck.motor_ids["pan"], "sts3215", MotorNormMode.RANGE_M100_100),
                "neck_nod_left": Motor(
                    config.neck.motor_ids["nod_left"], "sts3215", MotorNormMode.RANGE_M100_100
                ),
                "neck_nod_right": Motor(
                    config.neck.motor_ids["nod_right"], "sts3215", MotorNormMode.RANGE_M100_100
                ),
            }

        ids = [motor.id for motor in feetech_motors.values()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"ELO Feetech motor IDs must be unique on one bus. Got: {ids}")

        feetech_calibration = {k: v for k, v in self.calibration.items() if k in feetech_motors}
        self.feetech_bus = FeetechMotorsBus(
            port=config.feetech_port,
            motors=feetech_motors,
            calibration=feetech_calibration,
        )

        left_odrive_calibration = {k: v for k, v in self.calibration.items() if k == "left_joint_1"}
        right_odrive_calibration = {k: v for k, v in self.calibration.items() if k == "right_joint_1"}
        self.left_odrive_bus = ODriveMotorsBus(
            port=config.left_odrive_port,
            motors={"left_joint_1": Motor(config.left_odrive_axis, "gim6010-8", norm_mode_body)},
            calibration=left_odrive_calibration,
        )
        self.right_odrive_bus = ODriveMotorsBus(
            port=config.right_odrive_port,
            motors={"right_joint_1": Motor(config.right_odrive_axis, "gim6010-8", norm_mode_body)},
            calibration=right_odrive_calibration,
        )

        self.bus = SimpleNamespace(motors={})
        self.bus.motors.update(self.left_odrive_bus.motors)
        self.bus.motors.update(self.right_odrive_bus.motors)
        self.bus.motors.update(self.feetech_bus.motors)

        self.cameras = make_cameras_from_configs(config.cameras)
        self._neck_action = {"neck_yaw.pos": 0.0, "neck_pitch.pos": 0.0, "neck_roll.pos": 0.0}

    @property
    def _motors_ft(self) -> dict[str, type]:
        features = {"left_joint_1.pos": float, "right_joint_1.pos": float}
        features.update({f"{motor}.pos": float for motor in self.feetech_bus.motors})
        if self.config.neck is not None:
            features |= {"neck_yaw.pos": float, "neck_pitch.pos": float, "neck_roll.pos": float}
        return features

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
            and self.left_odrive_bus.is_connected
            and self.right_odrive_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.feetech_bus.connect()
        self.left_odrive_bus.connect()
        self.right_odrive_bus.connect()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration "
                "file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        if calibrate:
            self.configure()
        else:
            self.feetech_bus.disable_torque()
            self.left_odrive_bus.disable_torque()
            self.right_odrive_bus.disable_torque()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return (
            self.feetech_bus.is_calibrated
            and self.left_odrive_bus.is_calibrated
            and self.right_odrive_bus.is_calibrated
        )

    def _split_calibration(
        self,
    ) -> tuple[dict[str, MotorCalibration], dict[str, MotorCalibration], dict[str, MotorCalibration]]:
        feetech_names = set(self.feetech_bus.motors)
        feetech = {k: v for k, v in self.calibration.items() if k in feetech_names}
        left = {k: v for k, v in self.calibration.items() if k == "left_joint_1"}
        right = {k: v for k, v in self.calibration.items() if k == "right_joint_1"}
        return feetech, left, right

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Writing calibration file associated with the id %s to the motors", self.id)
                feetech_calibration, left_calibration, right_calibration = self._split_calibration()
                self.feetech_bus.write_calibration(feetech_calibration)
                self.left_odrive_bus.write_calibration(left_calibration)
                self.right_odrive_bus.write_calibration(right_calibration)
                self._save_calibration()
                return

        logger.info("\nRunning calibration of %s", self)
        self.left_odrive_bus.disable_torque()
        self.right_odrive_bus.disable_torque()
        self.feetech_bus.disable_torque()

        for motor in self.feetech_bus.motors:
            self.feetech_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        homing_offsets = {}
        odrive_zero_positions = {}
        for side, odrive_bus in (
            ("left", self.left_odrive_bus),
            ("right", self.right_odrive_bus),
        ):
            input(
                f"\nCalibration: Set {side.upper()} GEM arm zero position\n"
                f"Position only the {side} GEM arm in the usual calibration pose:\n"
                "  - Joints 1-3 hanging downward\n"
                "  - Joint 4 slightly bent (about 20 degrees below logical zero)\n"
                "  - Joint 5 centered\n"
                "  - Joint 6 pushed up to its upper limit so the hand bends upward\n"
                "  - Gripper closed\n"
                "Press ENTER when ready..."
            )

            joint_1 = f"{side}_joint_1"
            previous_odrive_calibration = odrive_bus.read_calibration()
            odrive_bus.write_calibration({}, cache=True)
            try:
                odrive_zero_positions[joint_1] = float(odrive_bus.read("Present_Position", joint_1))
            finally:
                odrive_bus.write_calibration(previous_odrive_calibration, cache=True)

            side_motors = [f"{side}_{joint}" for joint in _ARM_JOINT_MODELS] + [f"{side}_gripper"]
            side_offsets = self.feetech_bus.set_half_turn_homings(side_motors)
            side_offsets = self._apply_captured_pose_biases(side_offsets)
            for motor_name, offset in side_offsets.items():
                self.feetech_bus.write("Homing_Offset", motor_name, offset)
            homing_offsets.update(side_offsets)

        neck_motors = [motor for motor in self.feetech_bus.motors if motor.startswith("neck_")]
        if neck_motors:
            input("\nCalibration: Set ELO neck neutral position and press ENTER...")
            neck_offsets = self.feetech_bus.set_half_turn_homings(neck_motors)
            for motor_name, offset in neck_offsets.items():
                self.feetech_bus.write("Homing_Offset", motor_name, offset)
            homing_offsets.update(neck_offsets)

        range_mins: dict[str, int] = {}
        range_maxes: dict[str, int] = {}
        for motor_name, m in self.feetech_bus.motors.items():
            max_res = self.feetech_bus.model_resolution_table[m.model] - 1
            range_mins[motor_name] = 0
            range_maxes[motor_name] = max_res

        for gripper in ("left_gripper", "right_gripper"):
            input(f"\n{gripper} calibration: Move to desired 0% position and press ENTER...")
            gripper_zero = int(self.feetech_bus.read("Present_Position", gripper, normalize=False))
            input(f"{gripper} calibration: Move to desired 100% position and press ENTER...")
            gripper_max = int(self.feetech_bus.read("Present_Position", gripper, normalize=False))
            if gripper_zero == gripper_max:
                raise ValueError(f"Invalid {gripper} calibration: 0% and 100% positions are identical.")
            range_mins[gripper] = min(gripper_zero, gripper_max)
            range_maxes[gripper] = max(gripper_zero, gripper_max)

        self.calibration = {}
        for motor_name, m in self.feetech_bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=m.id,
                drive_mode=1 if motor_name.endswith("gripper") else 0,
                homing_offset=homing_offsets[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )

        self.calibration["left_joint_1"] = MotorCalibration(
            id=self.left_odrive_bus.motors["left_joint_1"].id,
            drive_mode=0,
            homing_offset=int(round(odrive_zero_positions["left_joint_1"])),
            range_min=-180,
            range_max=180,
        )
        self.calibration["right_joint_1"] = MotorCalibration(
            id=self.right_odrive_bus.motors["right_joint_1"].id,
            drive_mode=0,
            homing_offset=int(round(odrive_zero_positions["right_joint_1"])),
            range_min=-180,
            range_max=180,
        )

        feetech_calibration, left_calibration, right_calibration = self._split_calibration()
        self.feetech_bus.write_calibration(feetech_calibration)
        self.left_odrive_bus.write_calibration(left_calibration)
        self.right_odrive_bus.write_calibration(right_calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def _apply_captured_pose_biases(self, homing_offsets: dict[str, int]) -> dict[str, int]:
        adjusted_offsets = homing_offsets.copy()
        for side in ("left", "right"):
            for joint, captured_pose_deg in _CAPTURED_POSE_DEGREES.items():
                motor_name = f"{side}_{joint}"
                if motor_name not in adjusted_offsets:
                    continue
                model = self.feetech_bus.motors[motor_name].model
                max_res = self.feetech_bus.model_resolution_table[model] - 1
                ticks = int(round(captured_pose_deg * max_res / 360))
                adjusted_offsets[motor_name] -= ticks
        return adjusted_offsets

    def configure(self) -> None:
        with self.feetech_bus.torque_disabled():
            self.feetech_bus.configure_motors()
            for motor in self.feetech_bus.motors:
                self.feetech_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                if motor.startswith("neck_"):
                    continue
                self.feetech_bus.write("P_Coefficient", motor, 16)
                self.feetech_bus.write("I_Coefficient", motor, 0)
                self.feetech_bus.write("D_Coefficient", motor, 32)

        self.left_odrive_bus.configure_motors()
        self.right_odrive_bus.configure_motors()
        self.left_odrive_bus.enable_torque()
        self.right_odrive_bus.enable_torque()
        self._send_neck_pose(0.0, 0.0, 0.0)

    def setup_motors(self) -> None:
        for motor in reversed(self.feetech_bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.feetech_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.feetech_bus.motors[motor].id}")
        print("ODrive motor setup is not automated. Configure each SteadyWin ODrive board manually.")

    def _read_joint_positions(self) -> dict[str, float]:
        positions = self.feetech_bus.sync_read("Present_Position")
        positions["left_joint_1"] = float(self.left_odrive_bus.read("Present_Position", "left_joint_1"))
        positions["right_joint_1"] = float(self.right_odrive_bus.read("Present_Position", "right_joint_1"))
        return positions

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict = {f"{motor}.pos": val for motor, val in self._read_joint_positions().items()}
        if self.config.neck is not None:
            obs_dict.update(self._neck_action)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.read_latest()
        return obs_dict

    def _send_neck_pose(self, yaw: float, pitch: float, roll: float) -> None:
        if self.config.neck is None:
            return
        if abs(yaw) < self.config.neck.deadzone_degrees:
            yaw = 0.0
        if abs(pitch) < self.config.neck.deadzone_degrees:
            pitch = 0.0
        if abs(roll) < self.config.neck.deadzone_degrees:
            roll = 0.0

        pan_steps = _deg_to_steps(yaw, self.config.neck.pan_limit)
        nod_steps = _deg_to_steps(pitch, self.config.neck.tilt_limit)
        tilt_steps = _deg_to_steps(roll, self.config.neck.tilt_limit)
        self.feetech_bus.sync_write(
            "Goal_Position",
            {
                "neck_pan": self.config.neck.centers["pan"] + pan_steps,
                "neck_nod_left": self.config.neck.centers["nod_left"] + nod_steps + tilt_steps,
                "neck_nod_right": self.config.neck.centers["nod_right"] - nod_steps + tilt_steps,
            },
            normalize=False,
        )
        self._neck_action = {
            "neck_yaw.pos": yaw,
            "neck_pitch.pos": pitch,
            "neck_roll.pos": roll,
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        neck_keys = {"neck_yaw", "neck_pitch", "neck_roll"}
        motor_goal_pos = {key: val for key, val in goal_pos.items() if key not in neck_keys}

        if self.config.max_relative_target is not None and motor_goal_pos:
            present_pos = self._read_joint_positions()
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in motor_goal_pos.items()}
            motor_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        if "left_joint_1" in motor_goal_pos:
            self.left_odrive_bus.sync_write("Goal_Position", {"left_joint_1": motor_goal_pos["left_joint_1"]})
        if "right_joint_1" in motor_goal_pos:
            self.right_odrive_bus.sync_write(
                "Goal_Position", {"right_joint_1": motor_goal_pos["right_joint_1"]}
            )

        feetech_goal_pos = {k: v for k, v in motor_goal_pos.items() if k in self.feetech_bus.motors}
        if feetech_goal_pos:
            self.feetech_bus.sync_write("Goal_Position", feetech_goal_pos)

        if self.config.neck is not None:
            self._send_neck_pose(
                float(goal_pos.get("neck_yaw", self._neck_action["neck_yaw.pos"])),
                float(goal_pos.get("neck_pitch", self._neck_action["neck_pitch.pos"])),
                float(goal_pos.get("neck_roll", self._neck_action["neck_roll.pos"])),
            )

        return {f"{motor}.pos": val for motor, val in motor_goal_pos.items()} | self._neck_action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._send_neck_pose(0.0, 0.0, 0.0)
        self.left_odrive_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.right_odrive_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.feetech_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)
