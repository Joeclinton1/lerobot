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

import pytest

from lerobot.teleoperators.bi_minion_arm.bi_minion_arm import (
    _get_gripper_calibration,
    _wrap_homing_offset,
)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-2342, 1754),
        (2342, -1754),
        (-2047, -2047),
        (2047, 2047),
        (-2048, -2047),
        (2048, 2047),
    ],
)
def test_wrap_homing_offset(offset: int, expected: int) -> None:
    assert _wrap_homing_offset(offset) == expected


@pytest.mark.parametrize(
    ("zero_position", "hundred_position", "expected"),
    [
        (1419, 2047, (0, 1419, 2047)),
        (2047, 1419, (1, 1419, 2047)),
    ],
)
def test_get_gripper_calibration(
    zero_position: int, hundred_position: int, expected: tuple[int, int, int]
) -> None:
    assert _get_gripper_calibration("right_gripper", zero_position, hundred_position) == expected


def test_get_gripper_calibration_rejects_equal_points() -> None:
    with pytest.raises(ValueError, match="0% and 100% points are both 2047"):
        _get_gripper_calibration("right_gripper", 2047, 2047)
