# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction
from lerobot.utils.rotation import Rotation

from .config_phone import PhoneOS


def _axis_index(axis: str) -> int:
    indices = {"x": 0, "y": 1, "z": 2}
    try:
        return indices[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported phone target axis {axis!r}; expected one of x, y, z.") from exc


def _target_mapping_matrix(
    target_x_axis: str,
    target_y_axis: str,
    target_z_axis: str,
    target_x_sign: float,
    target_y_sign: float,
    target_z_sign: float,
) -> np.ndarray:
    mapping = np.zeros((3, 3), dtype=float)
    for row, (axis, sign) in enumerate(
        (
            (target_x_axis, target_x_sign),
            (target_y_axis, target_y_sign),
            (target_z_axis, target_z_sign),
        )
    ):
        mapping[row, _axis_index(axis)] = float(sign)

    if not np.isclose(np.linalg.det(mapping), 1.0):
        raise ValueError("Phone orientation axis mapping must be a right-handed rotation basis.")
    return mapping


@ProcessorStepRegistry.register("map_phone_action_to_robot_action")
@dataclass
class MapPhoneActionToRobotAction(RobotActionProcessorStep):
    """
    Maps calibrated phone pose actions to standardized robot action inputs.

    This processor step acts as a bridge between the phone teleoperator's output
    and the robot's expected action format. It remaps the phone's 6-DoF pose
    (position and rotation) to the robot's target end-effector pose, applying
    necessary axis inversions and swaps. It also interprets platform-specific
    button presses to generate a gripper command.

    Attributes:
        platform: The operating system of the phone (iOS or Android), used
            to determine the correct button mappings for the gripper.
    """

    # TODO(Steven): Gripper vel could be output of phone_teleop directly
    platform: PhoneOS
    use_so100_axis_mapping: bool = True
    target_x_axis: str = "x"
    target_y_axis: str = "y"
    target_z_axis: str = "z"
    target_x_sign: float = 1.0
    target_y_sign: float = 1.0
    target_z_sign: float = 1.0
    orientation_scale: float = 1.0

    def action(self, action: RobotAction) -> RobotAction:
        """
        Processes the phone action dictionary to create a robot action dictionary.

        Args:
            act: The input action dictionary from the phone teleoperator.

        Returns:
            A new action dictionary formatted for the robot controller.

        Raises:
            ValueError: If 'pos' or 'rot' keys are missing from the input action.
        """
        # Pop them from the action
        enabled = bool(action.pop("phone.enabled"))
        pos = action.pop("phone.pos")
        rot = action.pop("phone.rot")
        inputs = action.pop("phone.raw_inputs")

        if pos is None or rot is None:
            raise ValueError("pos and rot must be present in action")

        rotvec = rot.as_rotvec()  # Absolute orientation as rotvec

        # Map certain inputs to certain actions
        if self.platform == PhoneOS.IOS:
            gripper_vel = float(inputs.get("a3", 0.0))
        else:
            a = float(inputs.get("reservedButtonA", 0.0))
            b = float(inputs.get("reservedButtonB", 0.0))
            gripper_vel = (
                a - b
            )  # Positive if a is pressed, negative if b is pressed, 0 if both or neither are pressed

        action["enabled"] = enabled
        if self.use_so100_axis_mapping:
            action["target_x"] = -pos[1] if enabled else 0.0
            action["target_y"] = pos[0] if enabled else 0.0
            action["target_z"] = pos[2] if enabled else 0.0
            action["target_wx"] = rotvec[1] if enabled else 0.0
            action["target_wy"] = rotvec[0] if enabled else 0.0
            action["target_wz"] = -rotvec[2] if enabled else 0.0
        else:
            mapping = _target_mapping_matrix(
                self.target_x_axis,
                self.target_y_axis,
                self.target_z_axis,
                self.target_x_sign,
                self.target_y_sign,
                self.target_z_sign,
            )
            target_pos = mapping @ np.asarray(pos, dtype=float)
            target_rot = Rotation.from_matrix(mapping @ rot.as_matrix() @ mapping.T).as_rotvec()
            target_rot *= self.orientation_scale
            action["target_x"] = float(target_pos[0]) if enabled else 0.0
            action["target_y"] = float(target_pos[1]) if enabled else 0.0
            action["target_z"] = float(target_pos[2]) if enabled else 0.0
            action["target_wx"] = float(target_rot[0]) if enabled else 0.0
            action["target_wy"] = float(target_rot[1]) if enabled else 0.0
            action["target_wz"] = float(target_rot[2]) if enabled else 0.0
        action["gripper_vel"] = gripper_vel  # Still send gripper action when disabled
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "pos", "rot", "raw_inputs"]:
            features[PipelineFeatureType.ACTION].pop(f"phone.{feat}", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
