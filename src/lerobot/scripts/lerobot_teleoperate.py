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

"""
Simple script to control a robot from teleoperation.

Requires: pip install 'lerobot[hardware]'

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import TYPE_CHECKING

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_gem_follower,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    gem_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)

# Import config modules to register supported robot and teleoperator subclasses with draccus.
from lerobot.robots.bi_gem_follower import config_bi_gem_follower  # noqa: F401
from lerobot.robots.gem_follower import config_gem_follower  # noqa: F401
from lerobot.robots.none_robot import config_none_robot  # noqa: F401
from lerobot.robots.none_robot.config_none_robot import NoneRobotConfig
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_minion_arm,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    hand_teleop,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    minion_arm,
    omx_leader,
    openarm_leader,
    openarm_mini,
    phone,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.bi_minion_arm import config_bi_minion_arm  # noqa: F401
from lerobot.teleoperators.minion_arm import config_minion_arm  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_arm_viewer import RobotArmViewer, RobotArmViewerConfig, map_action_to_gem
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

if TYPE_CHECKING:
    from lerobot.processor import RobotProcessorPipeline


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    teleop_calibrate: bool = True
    robot_calibrate: bool = True
    # Optional robot-arm-viewer sidecar for visualizing GEM/BiGEM actions.
    viewer: RobotArmViewerConfig = field(default_factory=RobotArmViewerConfig)


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: "RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]",
    robot_action_processor: "RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]",
    robot_observation_processor: "RobotProcessorPipeline[RobotObservation, RobotObservation]",
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    viewer: RobotArmViewer | None = None,
    viewer_initial_action: RobotAction | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max((len(key) for key in robot.action_features), default=1)
    viewer_action_state: RobotAction | None = (
        None if viewer_initial_action is None else dict(viewer_initial_action)
    )
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()
        if not obs and viewer_action_state is not None:
            obs = dict(viewer_action_state)

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        # Get teleop action. Hand teleop uses current joint state as its IK seed.
        raw_action = _get_teleop_action(teleop, obs, viewer_action_state if robot.name == "none" else None)
        if robot.name == "none" and _is_hand_teleop(teleop):
            viewer_action_state = dict(raw_action)

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        if robot.name in {"gem", "bi_gem"}:
            robot_action_to_send = map_action_to_gem(robot_action_to_send)

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        sent_action = robot.send_action(robot_action_to_send)
        if robot.name == "none" and _is_joint_position_action(sent_action):
            viewer_action_state = dict(sent_action)
        if viewer is not None:
            viewer.send_action(raw_action if _is_bimanual_action(raw_action) else sent_action)

        if display_data:
            from lerobot.utils.visualization_utils import log_rerun_data

            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    if cfg.viewer.only and not cfg.viewer.enabled:
        raise ValueError("--viewer.only=true requires --viewer.enabled=true")

    if _is_hand_teleop_config(cfg.teleop):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        _make_teleoperate_processors(cfg)
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(NoneRobotConfig()) if cfg.viewer.only else make_robot_from_config(cfg.robot)
    viewer = make_robot_arm_viewer(cfg.viewer, cfg.robot.type)
    viewer_initial_action = _viewer_initial_action(cfg)

    teleop.connect(calibrate=cfg.teleop_calibrate)
    if (
        cfg.viewer.only
        and _is_hand_teleop(teleop)
        and cfg.robot.type in {"gem", "gem_follower", "bi_gem", "bi_gem_follower"}
    ):
        align_to_viewer = getattr(teleop, "align_kinematics_to_gem_viewer", None)
        if align_to_viewer is not None:
            align_to_viewer()
    robot.connect(calibrate=cfg.robot_calibrate)
    if viewer is not None:
        viewer.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            viewer=viewer,
            viewer_initial_action=viewer_initial_action,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            from lerobot.utils.visualization_utils import shutdown_rerun

            shutdown_rerun()
        teleop.disconnect()
        robot.disconnect()
        if viewer is not None:
            viewer.disconnect()


def make_robot_arm_viewer(config: RobotArmViewerConfig, robot_type: str) -> RobotArmViewer | None:
    if not config.enabled:
        return None
    return RobotArmViewer(config, robot_type)


def _is_bimanual_action(action: RobotAction) -> bool:
    return any(key.startswith(("left_", "right_")) for key in action)


def _is_joint_position_action(action: RobotAction) -> bool:
    return any(key.endswith(".pos") for key in action)


def _is_hand_teleop(teleop: Teleoperator) -> bool:
    return getattr(teleop, "name", None) in {"hand_teleop", "handteleop"} or teleop.__class__.__name__ == "HandTeleop"


def _is_hand_teleop_config(teleop_config: TeleoperatorConfig) -> bool:
    return getattr(teleop_config, "type", None) in {"hand_teleop", "handteleop"}


def _is_phone_teleop_config(teleop_config: TeleoperatorConfig) -> bool:
    return getattr(teleop_config, "type", None) == "phone"


def _is_single_gem_robot_config(robot_config: RobotConfig) -> bool:
    return getattr(robot_config, "type", None) in {"gem", "gem_follower"}


def _is_gem_robot_config(robot_config: RobotConfig) -> bool:
    return getattr(robot_config, "type", None) in {"gem", "gem_follower", "bi_gem", "bi_gem_follower"}


def _make_teleoperate_processors(cfg: TeleoperateConfig):
    if _is_phone_teleop_config(cfg.teleop) and _is_gem_robot_config(cfg.robot):
        return _make_phone_to_gem_processors(cfg.teleop, cfg.robot)

    from lerobot.processor import make_default_processors

    return make_default_processors()


def _make_phone_to_gem_processors(teleop_config: TeleoperatorConfig, robot_config: RobotConfig):
    from lerobot.processor import (
        RobotProcessorPipeline,
        make_default_robot_observation_processor,
        robot_action_observation_to_transition,
        transition_to_robot_action,
    )
    from lerobot.robots.gem_follower.kinematics import GEM_MOTOR_NAMES, make_gem_kinematics
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        EEBoundsAndSafety,
        EEReferenceAndDelta,
        GripperVelocityToJoint,
        InverseKinematicsEEToJoints,
        JointRateLimit,
    )
    from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction

    arm = getattr(getattr(teleop_config, "arm", "left"), "value", getattr(teleop_config, "arm", "left"))
    if arm not in {"left", "right"}:
        raise ValueError(f"Unsupported phone arm {arm!r}; expected 'left' or 'right'.")

    target_axes = (
        getattr(teleop_config, "target_x_axis", "x"),
        getattr(teleop_config, "target_y_axis", "y"),
        getattr(teleop_config, "target_z_axis", "z"),
    )
    target_signs = (
        float(getattr(teleop_config, "target_x_sign", 1.0)),
        float(getattr(teleop_config, "target_y_sign", 1.0)),
        float(getattr(teleop_config, "target_z_sign", 1.0)),
    )
    position_scale = float(getattr(teleop_config, "position_scale", 0.5))
    orientation_scale = float(getattr(teleop_config, "orientation_scale", 1.0))
    position_weight = float(getattr(teleop_config, "position_weight", 200.0))
    orientation_weight = float(getattr(teleop_config, "orientation_weight", 4.0))
    max_joint_step_deg = float(getattr(teleop_config, "max_joint_step_deg", 3.0))

    is_bimanual_gem = getattr(robot_config, "type", None) in {"bi_gem", "bi_gem_follower"}
    motor_names = [f"{arm}_{name}" for name in GEM_MOTOR_NAMES] if is_bimanual_gem else GEM_MOTOR_NAMES
    gripper_name = f"{arm}_gripper" if is_bimanual_gem else "gripper"

    kinematics = make_gem_kinematics(arm=arm)
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapPhoneActionToRobotAction(
                platform=teleop_config.phone_os,
                use_so100_axis_mapping=False,
                target_x_axis=target_axes[0],
                target_y_axis=target_axes[1],
                target_z_axis=target_axes[2],
                target_x_sign=target_signs[0],
                target_y_sign=target_signs[1],
                target_z_sign=target_signs[2],
                orientation_scale=orientation_scale,
            ),
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes={"x": position_scale, "y": position_scale, "z": position_scale},
                motor_names=motor_names,
                use_latched_reference=True,
                orientation_delta_in_world=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.05,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
                clip_min=0.0,
                clip_max=100.0,
                gripper_name=gripper_name,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=False,
                iterations=5,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
                gripper_name=gripper_name,
                max_position_error_m=0.05,
                # The kinematics posture task now resolves the redundant elbow deterministically,
                # so the multi-seed elbow search is no longer needed.
                joint_selection_weights=[0.5, 1.0, 4.0, 2.0, 1.5, 1.5, 1.0, 0.0],
            ),
            # Final safety net: bound per-step joint motion so tracking jumps or elbow branch
            # changes ramp smoothly instead of snapping. Runs after IK on the actual joint commands.
            JointRateLimit(
                motor_names=motor_names,
                max_joint_step_deg=max_joint_step_deg,
                gripper_name=gripper_name,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor, robot_action_processor, make_default_robot_observation_processor()


def _viewer_initial_action(cfg: TeleoperateConfig) -> RobotAction | None:
    if not (
        cfg.viewer.only and _is_phone_teleop_config(cfg.teleop) and _is_gem_robot_config(cfg.robot)
    ):
        return None

    from lerobot.robots.gem_follower.kinematics import GEM_MOTOR_NAMES

    arm = getattr(getattr(cfg.teleop, "arm", "left"), "value", getattr(cfg.teleop, "arm", "left"))
    prefix = f"{arm}_" if getattr(cfg.robot, "type", None) in {"bi_gem", "bi_gem_follower"} else ""
    return {f"{prefix}{name}.pos": 5.0 if name == "gripper" else 0.0 for name in GEM_MOTOR_NAMES}


def _get_teleop_action(
    teleop: Teleoperator,
    obs: RobotObservation,
    fallback_state: RobotAction | None,
) -> RobotAction:
    if not _is_hand_teleop(teleop):
        return teleop.get_action()

    current_state = _current_state_for_hand_teleop(teleop, obs, fallback_state)
    if current_state is None:
        return teleop.get_action()
    return teleop.get_action(current_state)


def _current_state_for_hand_teleop(
    teleop: Teleoperator,
    obs: RobotObservation,
    fallback_state: RobotAction | None,
) -> RobotAction | None:
    action_features = getattr(teleop, "action_features", {})
    if action_features and all(key in obs for key in action_features):
        return {key: float(obs[key]) for key in action_features}
    return fallback_state


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
