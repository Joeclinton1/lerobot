from __future__ import annotations

from types import SimpleNamespace

from lerobot.motors.motors_bus import Motor, MotorNormMode
from lerobot.motors.odrive.odrive import ODriveMotorsBus


class _CountingDescriptor:
    def __init__(self, value: float):
        self.value = value
        self.read_count = 0

    def __get__(self, instance, owner):
        if instance is None:
            return self
        self.read_count += 1
        return self.value


def _make_bus() -> ODriveMotorsBus:
    motors = {
        "joint_a": Motor(id=0, model="gim6010-8", norm_mode=MotorNormMode.DEGREES),
        "joint_b": Motor(id=1, model="gim6010-8", norm_mode=MotorNormMode.DEGREES),
    }
    return ODriveMotorsBus(port="auto", motors=motors)


def test_axis_position_turns_only_reads_descriptor_once():
    bus = _make_bus()
    descriptor = _CountingDescriptor(1.25)
    estimator_type = type("Estimator", (), {"pos_estimate": descriptor})
    axis = SimpleNamespace(encoder_estimator=estimator_type())

    assert bus._axis_position_turns(axis) == 1.25
    assert descriptor.read_count == 1
