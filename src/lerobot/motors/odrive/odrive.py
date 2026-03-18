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

from __future__ import annotations

import logging
from importlib import import_module
from contextlib import contextmanager
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _odrive_available

if TYPE_CHECKING or _odrive_available:
    import odrive
else:
    odrive = SimpleNamespace()  # type: ignore[assignment]

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value

logger = logging.getLogger(__name__)

_LEGACY_AXIS_STATES = {
    "IDLE": "AXIS_STATE_IDLE",
    "CLOSED_LOOP_CONTROL": "AXIS_STATE_CLOSED_LOOP_CONTROL",
}

_LEGACY_CONTROL_MODES = {
    "POSITION_CONTROL": "CONTROL_MODE_POSITION_CONTROL",
}

_LEGACY_INPUT_MODES = {
    "PASSTHROUGH": "INPUT_MODE_PASSTHROUGH",
}

# Gear reduction defaults for Steadywin GIM6010 variants.
# ODrive position units are turns, so we divide/multiply by this ratio when exposing degrees.
_MODEL_GEAR_RATIO_DEFAULTS = {
    "gim6010": 8.0,
    "gim6010-6": 6.0,
    "gim6010_6": 6.0,
    "gim6010-8": 8.0,
    "gim6010_8": 8.0,
    "gim6010-10": 10.0,
    "gim6010_10": 10.0,
    "gim6010-36": 36.0,
    "gim6010_36": 36.0,
    "gim6010-48": 48.0,
    "gim6010_48": 48.0,
}


def _infer_gear_ratio_from_model(model: str) -> float:
    model_l = model.strip().lower()
    if model_l in _MODEL_GEAR_RATIO_DEFAULTS:
        return _MODEL_GEAR_RATIO_DEFAULTS[model_l]

    # Common vendor naming (e.g. gim6010-8-lite / gim6010_lite_8).
    match = re.search(r"gim6010(?:[-_]?lite)?[-_](\d+)", model_l)
    if match:
        return float(match.group(1))

    return 1.0


class ODriveMotorsBus(MotorsBusBase):
    """
    ODrive motor bus implementation based on the pip-installable `odrive` Python package.

    This bus maps each configured motor to one ODrive axis (`axis{motor.id}`), where `motor.id`
    must be 0 or 1 for a single dual-axis ODrive board.
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        connection_timeout_s: float = 20.0,
    ):
        super().__init__(port, motors, calibration)
        self.connection_timeout_s = connection_timeout_s
        self._odrive_device: Any | None = None
        self._is_connected = False
        self._gains: dict[str, dict[str, float]] = {name: {"kp": 0.0, "kd": 0.0} for name in self.motors}
        self._gear_ratios: dict[str, float] = {}
        for name, motor in self.motors.items():
            ratio = float(_infer_gear_ratio_from_model(motor.model))
            if ratio <= 0:
                raise ValueError(f"Invalid gear ratio for motor '{name}': {ratio}. Expected ratio > 0.")
            self._gear_ratios[name] = ratio

    def _get_gear_ratio(self, motor: str) -> float:
        return self._gear_ratios.get(motor, 1.0)

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._odrive_device is not None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        if not _odrive_available:
            raise ImportError(
                "ODrive support requires the `odrive` package. Install it with `pip install odrive` "
                "or `pip install -e \".[odrive]\"`."
            )

        serial_number = self.port if self.port not in ("", "any", "auto") else None

        find_sync = getattr(odrive, "find_sync", None)
        find_any = getattr(odrive, "find_any", None)

        kwargs = {"timeout": self.connection_timeout_s}
        if serial_number is not None:
            kwargs["serial_number"] = serial_number

        if callable(find_sync):
            self._odrive_device = find_sync(**kwargs)
        elif callable(find_any):
            self._odrive_device = find_any(**kwargs)
        else:
            raise RuntimeError("ODrive Python API does not expose `find_sync` or `find_any`.")

        if self._odrive_device is None:
            raise ConnectionError(
                "Failed to find an ODrive device. Check USB connection and `port`/serial-number config."
            )

        self._is_connected = True

        if handshake:
            self._handshake()

        logger.debug(f"{self.__class__.__name__} connected.")

    def _handshake(self) -> None:
        missing_axes = []
        for motor in self.motors:
            try:
                self._get_axis(motor)
            except Exception:
                missing_axes.append(motor)

        if missing_axes:
            raise ConnectionError(
                f"ODrive handshake failed. Missing axes for motors: {missing_axes}. "
                "Check that motor IDs map to valid axis numbers."
            )

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.disable_torque()
        self._odrive_device = None
        self._is_connected = False
        logger.debug(f"{self.__class__.__name__} disconnected.")

    def configure_motors(self, motors: str | list[str] | None = None) -> None:
        """Configure target axes in position-control + passthrough mode when available."""
        for motor in self._get_motors_list(motors):
            axis = self._get_axis(motor)
            self._configure_axis(axis)

    def _get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        if isinstance(motor, int):
            return motor
        raise TypeError(f"Unsupported motor selector type: {type(motor)}")

    def _get_motor_name(self, motor: NameOrID) -> str:
        if isinstance(motor, str):
            return motor
        for name, m in self.motors.items():
            if m.id == motor:
                return name
        raise KeyError(f"No motor mapped to id={motor}.")

    def _get_motors_list(self, motors: NameOrID | list[NameOrID] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        if isinstance(motors, (str, int)):
            return [self._get_motor_name(motors)]
        if isinstance(motors, list):
            return [self._get_motor_name(m) for m in motors]
        raise TypeError(f"Unsupported motors spec type: {type(motors)}")

    def _get_axis(self, motor: NameOrID) -> Any:
        if self._odrive_device is None:
            raise RuntimeError("ODrive device not connected.")

        axis_id = self._get_motor_id(motor)
        axis_name = f"axis{axis_id}"
        axis = getattr(self._odrive_device, axis_name, None)
        if axis is None:
            raise KeyError(f"ODrive has no `{axis_name}`.")
        return axis

    def _get_enum_value(self, enum_class_name: str, member_name: str, legacy_name: str) -> Any:
        enum_class = getattr(odrive, enum_class_name, None)
        if enum_class is not None:
            member = getattr(enum_class, member_name, None)
            if member is not None:
                return member

        enums_module = getattr(odrive, "enums", None)
        if enums_module is not None and hasattr(enums_module, legacy_name):
            return getattr(enums_module, legacy_name)

        # Some odrive versions expose enums only as a separate submodule (not bound to odrive.enums attr).
        try:
            enums_module = import_module("odrive.enums")
            if hasattr(enums_module, legacy_name):
                return getattr(enums_module, legacy_name)
        except Exception:
            pass

        if hasattr(odrive, legacy_name):
            return getattr(odrive, legacy_name)

        raise RuntimeError(f"Unable to resolve ODrive enum value for {enum_class_name}.{member_name}.")

    def _request_state(self, axis: Any, state: Any) -> None:
        utils_module = getattr(odrive, "utils", None)
        request_state = getattr(utils_module, "request_state", None) if utils_module else None
        if callable(request_state):
            request_state(axis, state)
        else:
            axis.requested_state = int(state)

    def _configure_axis(self, axis: Any) -> None:
        controller = getattr(axis, "controller", None)
        if controller is None:
            logger.warning("ODrive axis has no controller attribute — skipping configuration.")
            return

        config = getattr(controller, "config", None)
        if config is None:
            logger.warning("ODrive controller has no config attribute — skipping configuration.")
            return

        try:
            control_mode = self._get_enum_value(
                "ControlMode", "POSITION_CONTROL", _LEGACY_CONTROL_MODES["POSITION_CONTROL"]
            )
            config.control_mode = int(control_mode)
        except Exception as e:
            logger.warning(f"Failed to set ODrive control mode to POSITION_CONTROL: {e}")

        try:
            input_mode = self._get_enum_value("InputMode", "PASSTHROUGH", _LEGACY_INPUT_MODES["PASSTHROUGH"])
            config.input_mode = int(input_mode)
        except Exception as e:
            logger.warning(f"Failed to set ODrive input mode to PASSTHROUGH: {e}")

    def _axis_position_turns(self, axis: Any) -> float:
        # Prefer actual encoder estimates over commanded input_pos.
        candidates = [
            ("encoder_estimator", "pos_estimate"),
            ("encoder", "pos_estimate"),
            ("pos_vel_mapper", "pos_rel"),
            ("pos_vel_mapper", "pos_abs"),
        ]
        for obj_name, attr_name in candidates:
            obj = getattr(axis, obj_name, None)
            if obj is not None and hasattr(obj, attr_name):
                return float(getattr(obj, attr_name))

        # Last resort: read back the commanded position (not actual).
        controller = getattr(axis, "controller", None)
        if controller is not None and hasattr(controller, "pos_setpoint"):
            return float(controller.pos_setpoint)
        if controller is not None and hasattr(controller, "input_pos"):
            return float(controller.input_pos)

        raise ValueError("Unable to read axis position estimate from ODrive API.")

    def _axis_velocity_turns_per_second(self, axis: Any) -> float:
        candidates = [
            ("encoder_estimator", "vel_estimate"),
            ("encoder", "vel_estimate"),
            ("pos_vel_mapper", "vel"),
        ]
        for obj_name, attr_name in candidates:
            obj = getattr(axis, obj_name, None)
            if obj is not None and hasattr(obj, attr_name):
                return float(getattr(obj, attr_name))
        raise ValueError("Unable to read axis velocity estimate from ODrive API.")

    def _turns_to_output(self, motor: str, turns_value: float) -> float:
        if self.motors[motor].norm_mode.value == "degrees":
            return (turns_value * 360.0) / self._get_gear_ratio(motor)
        return turns_value

    def _output_to_turns(self, motor: str, value: float) -> float:
        if self.motors[motor].norm_mode.value == "degrees":
            return (value / 360.0) * self._get_gear_ratio(motor)
        return value

    def _get_motor_calibration(self, motor: str) -> MotorCalibration | None:
        return self.calibration.get(motor)

    def _calibrate_output_value(self, motor: str, raw_output_value: float) -> float:
        """Map raw axis output value to calibrated user-space value."""
        cal = self._get_motor_calibration(motor)
        if cal is None:
            return raw_output_value
        return raw_output_value - float(cal.homing_offset)

    def _uncalibrate_output_value(self, motor: str, calibrated_value: float) -> float:
        """Map calibrated user-space value back to raw axis output value."""
        cal = self._get_motor_calibration(motor)
        if cal is None:
            return calibrated_value

        # Clamp in calibrated (joint-space) coordinates, then shift back to raw.
        bounded_value = max(float(cal.range_min), min(float(cal.range_max), calibrated_value))
        raw_value = bounded_value + float(cal.homing_offset)
        return raw_value

    @check_if_not_connected
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        _ = num_retry
        closed_loop = self._get_enum_value(
            "AxisState", "CLOSED_LOOP_CONTROL", _LEGACY_AXIS_STATES["CLOSED_LOOP_CONTROL"]
        )
        for motor in self._get_motors_list(motors):
            axis = self._get_axis(motor)
            self._configure_axis(axis)
            self._request_state(axis, closed_loop)

    @check_if_not_connected
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        _ = num_retry
        idle = self._get_enum_value("AxisState", "IDLE", _LEGACY_AXIS_STATES["IDLE"])
        for motor in self._get_motors_list(motors):
            axis = self._get_axis(motor)
            self._request_state(axis, idle)

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    @check_if_not_connected
    def read(self, data_name: str, motor: str) -> Value:
        axis = self._get_axis(motor)
        if data_name == "Present_Position":
            raw_output = self._turns_to_output(motor, self._axis_position_turns(axis))
            return self._calibrate_output_value(motor, raw_output)
        if data_name == "Present_Velocity":
            return self._turns_to_output(motor, self._axis_velocity_turns_per_second(axis))
        if data_name == "Goal_Position":
            controller = getattr(axis, "controller", None)
            if controller is None or not hasattr(controller, "input_pos"):
                raise ValueError("ODrive controller input position is unavailable.")
            raw_output = self._turns_to_output(motor, float(controller.input_pos))
            return self._calibrate_output_value(motor, raw_output)
        if data_name == "Kp":
            return self._gains[motor]["kp"]
        if data_name == "Kd":
            return self._gains[motor]["kd"]

        raise ValueError(f"Unsupported read data_name: {data_name}")

    @check_if_not_connected
    def write(self, data_name: str, motor: str, value: Value) -> None:
        axis = self._get_axis(motor)

        if data_name == "Goal_Position":
            controller = getattr(axis, "controller", None)
            if controller is None:
                raise ValueError("ODrive axis controller is unavailable.")
            raw_output = self._uncalibrate_output_value(motor, float(value))
            target_turns = self._output_to_turns(motor, raw_output)
            controller.input_pos = target_turns
            return

        if data_name == "Kp":
            self._gains[motor]["kp"] = float(value)
            return

        if data_name == "Kd":
            self._gains[motor]["kd"] = float(value)
            return

        raise ValueError(f"Unsupported write data_name: {data_name}")

    @check_if_not_connected
    def sync_read(self, data_name: str, motors: NameOrID | list[NameOrID] | None = None) -> dict[str, Value]:
        return {motor: self.read(data_name, motor) for motor in self._get_motors_list(motors)}

    @check_if_not_connected
    def sync_write(self, data_name: str, values: Value | dict[str, Value]) -> None:
        if isinstance(values, dict):
            for motor, value in values.items():
                self.write(data_name, motor, value)
            return

        for motor in self.motors:
            self.write(data_name, motor, values)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.calibration.copy()

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        if cache:
            self.calibration = calibration_dict
