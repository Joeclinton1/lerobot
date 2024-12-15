"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


def ensure_safe_goal_position(
        goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos


@dataclass
class ManipulatorRobotConfig:
    """
    Example of usage:
    ```python
    ManipulatorRobotConfig()
    ```
    """

    # Define all components of the robot
    leader_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    SUPPORTED_ROBOT_TYPES = ["koch", "koch_bimanual", "aloha", "so100", "moss"]

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    def __setattr__(self, prop: str, val):
        if prop == "max_relative_target" and val is not None and isinstance(val, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(val):
                    raise ValueError(
                        f"len(max_relative_target)={len(val)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )
        super().__setattr__(prop, val)

    def __post_init__(self):
        # Check that each arm has a valid `robot_type` and `calibration_dir`
        for arm_name, arm in {**self.leader_arms, **self.follower_arms}.items():
            if not hasattr(arm, 'robot_type'):
                raise ValueError(f"The arm '{arm_name}' is missing the 'robot_type' attribute.")
            if not hasattr(arm, 'calibration_dir'):
                raise ValueError(f"The arm '{arm_name}' is missing the 'calibration_dir' attribute.")
            if arm.robot_type not in self.SUPPORTED_ROBOT_TYPES:
                raise ValueError(
                    f"Unsupported robot type '{arm.robot_type}' for arm '{arm_name}'. "
                    f"Supported types are: {', '.join(self.SUPPORTED_ROBOT_TYPES)}."
                )


class ManipulatorRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.

    Non exaustive list of robots:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) developed by Trossen Robotics

    Example of highest frequency teleoperation without camera:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                "shoulder_pan": (1, "xl330-m077"),
                # Other motors...
            },
            robot_type="koch",
            calibration_dir=Path(".cache/calibration/koch")
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                "shoulder_pan": (1, "xl430-w250"),
                # Other motors...
            },
            robot_type="koch",
            calibration_dir=Path(".cache/calibration/koch")
        ),
    }
    robot = ManipulatorRobot(
        leader_arms=leader_arms,
        follower_arms=follower_arms
    )

    # Connect motors buses and cameras if any (Required)
    robot.connect()

    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection without camera:
    ```python
    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
    )
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of highest frequency data collection with cameras:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }

    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
        cameras=cameras,
    )
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy (without running multiple policies in parallel to ensure highest frequency):
    ```python
    # Assumes leader and follower arms + cameras have been instantiated already (see previous example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
        cameras=cameras,
    )
    robot.connect()
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
            self,
            config: ManipulatorRobotConfig | None = None,
            **kwargs,
    ):
        if config is None:
            config = ManipulatorRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.leader_arms = self.config.leader_arms
        self.follower_arms = self.config.follower_arms
        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name, arm in self.follower_arms.items():
            print(f"Connecting {name} follower arm.")
            arm.connect()
        for name, arm in self.leader_arms.items():
            print(f"Connecting {name} leader arm.")
            arm.connect()

        # Dynamically select TorqueMode based on the robot type for each arm
        isFeetech = any([motor[1] in ['sts3215'] for motor in arm.motors.values()])
        isDynamixel = any([motor[1] in ['xl330-m077', 'xl330-m288', 'xl430-w250'] for motor in arm.motors.values()])
        if isDynamixel:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif isFeetech:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # Disable torque for safety during calibration or preset
        arm.write("Torque_Enable", TorqueMode.DISABLED.value)

        self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for specific robot types)
        for name, arm in self.leader_arms.items():
            if arm.robot_type in ["koch", "koch_bimanual"]:
                self.set_koch_robot_preset(arm)
            elif arm.robot_type == "aloha":
                self.set_aloha_robot_preset(arm)
            elif arm.robot_type in ["so100", "moss"]:
                self.set_so100_robot_preset(arm)

        # Enable torque on all motors of the follower arms
        for name, arm in self.follower_arms.items():
            print(f"Activating torque on {name} follower arm.")
            arm.write("Torque_Enable", 1)

        # Set gripper torque if configured
        if self.config.gripper_open_degree is not None:
            for name, arm in self.leader_arms.items():
                if arm.robot_type not in ["koch", "koch_bimanual"]:
                    raise NotImplementedError(
                        f"{arm.robot_type} does not support position AND current control in the handle, "
                        "which is required to set the gripper open."
                    )
                arm.write("Torque_Enable", 1, "gripper")
                arm.write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Check both arms can be read
        for name, arm in self.follower_arms.items():
            arm.read("Present_Position")
        for name, arm in self.leader_arms.items():
            arm.read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def activate_calibration(self):
        """After calibration all motors function in human-interpretable ranges."""

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = arm.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # Display a warning if calibration file is not available
                print(f"Missing calibration file '{arm_calib_path}'")

                # Run calibration based on robot type
                isFeetech = any([motor[1] in ['sts3215'] for motor in arm.motors.values()])
                isDynamixel = any([motor[1] in ['xl330-m077', 'xl330-m288', 'xl430-w250'] for motor in arm.motors.values()])

                print(arm.motors, isFeetech, isDynamixel)

                if isDynamixel:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration
                    calibration = run_arm_calibration(arm, arm.robot_type, name, arm_type)
                elif isFeetech:
                    from lerobot.common.robot_devices.robots.feetech_calibration import \
                        run_arm_manual_calibration
                    calibration = run_arm_manual_calibration(arm, arm.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)


    def set_koch_robot_preset(self, arm):
        """Set Koch robot preset configurations for a specific arm."""

        def set_operating_mode_(arm):
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

            # Use 'extended position mode' for all motors except gripper
            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                arm.write("Operating_Mode", 4, all_motors_except_gripper)

            # Set 'position control current based' mode for the gripper motor
            arm.write("Operating_Mode", 5, "gripper")

        # Apply the operating mode settings
        set_operating_mode_(arm)

        # Set better PID values for specific joints
        if "elbow_flex" in arm.motor_names:
            arm.write("Position_P_Gain", 1500, "elbow_flex")
            arm.write("Position_I_Gain", 0, "elbow_flex")
            arm.write("Position_D_Gain", 600, "elbow_flex")

        # Set gripper position if specified
        if self.config.gripper_open_degree is not None and "gripper" in arm.motor_names:
            arm.write("Torque_Enable", 1, "gripper")
            arm.write("Goal_Position", self.config.gripper_open_degree, "gripper")


    def set_aloha_robot_preset(self, arm):
        """Set Aloha robot preset configurations for a specific arm."""

        def set_shadow_(arm):
            # Set secondary/shadow ID for shoulder and elbow if dual-motor setup is present
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        # Apply shadow settings for dual-motor joints
        set_shadow_(arm)

        # Set a velocity limit if supported
        if "Velocity_Limit" in arm.motor_names:
            arm.write("Velocity_Limit", 131)

        # Set 'extended position mode' for all motors except the gripper
        all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
        if len(all_motors_except_gripper) > 0:
            arm.write("Operating_Mode", 4, all_motors_except_gripper)

        # Set 'position control current based' for the gripper if present
        if "gripper" in arm.motor_names:
            arm.write("Operating_Mode", 5, "gripper")

        # Warn if `gripper_open_degree` is set, as it's not supported for Aloha robots
        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree` is set to {self.config.gripper_open_degree}, "
                "but None is expected for Aloha instead.",
                stacklevel=1,
            )


    def set_so100_robot_preset(self, arm):
        """Set SO100 robot preset configurations for a specific arm."""
        # Set Mode to Position Control (Mode=0)
        arm.write("Mode", 0)

        # Set PID Coefficients to reduce shakiness
        arm.write("P_Coefficient", 16)
        arm.write("I_Coefficient", 0)
        arm.write("D_Coefficient", 32)

        # Close the write lock to ensure maximum acceleration is written to EPROM
        arm.write("Lock", 0)

        # Set maximum acceleration to enhance motor responsiveness
        arm.write("Maximum_Acceleration", 254)
        arm.write("Acceleration", 254)


    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
