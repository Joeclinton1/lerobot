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

"""
Print live joint tables for Minion leader, GEM follower, or both.
"""

import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.odrive import ODriveMotorsBus
from lerobot.teleoperators.minion_arm.config_minion_arm import MinionArmConfig
from lerobot.teleoperators.minion_arm.minion_arm import MinionArm
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

JOINT_ORDER = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
GEM_ODRIVE_AXIS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["leader", "follower", "both"], default="leader")
    parser.add_argument("--port", dest="leader_port", help="Leader serial port (e.g. COM7).")
    parser.add_argument("--id", dest="leader_id", default="minion_leader", help="Leader teleoperator id.")
    parser.add_argument("--leader-port", dest="leader_port", help="Leader serial port (e.g. COM7).")
    parser.add_argument("--leader-id", dest="leader_id", default="minion_leader", help="Leader teleoperator id.")
    parser.add_argument(
        "--leader-calibrate",
        action="store_true",
        help="Allow leader calibration flow if calibration is missing/mismatched.",
    )
    parser.add_argument("--follower-feetech-port", help="Follower Feetech serial port (e.g. COM8).")
    parser.add_argument("--follower-odrive-port", default="auto", help='Follower ODrive serial number or "auto".')
    parser.add_argument("--follower-id", default="gem_follower", help="Follower robot id for calibration lookup.")
    parser.add_argument(
        "--follower-raw",
        action="store_true",
        help="Read follower raw encoder values instead of calibrated values.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Refresh rate.")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear terminal each frame; print continuously instead.",
    )
    args = parser.parse_args()

    if args.mode in ("leader", "both") and not args.leader_port:
        parser.error("--leader-port (or --port) is required for mode leader/both.")
    if args.mode in ("follower", "both") and not args.follower_feetech_port:
        parser.error("--follower-feetech-port is required for mode follower/both.")

    return args


def _follower_calibration_path(robot_id: str) -> Path:
    return HF_LEROBOT_CALIBRATION / ROBOTS / "gem" / f"{robot_id}.json"


def _load_calibration(path: Path) -> dict[str, MotorCalibration]:
    with open(path) as f:
        data = json.load(f)
    return {name: MotorCalibration(**values) for name, values in data.items()}


def _connect_follower(args: argparse.Namespace) -> tuple[FeetechMotorsBus, ODriveMotorsBus]:
    feetech_motors = {
        "joint_2": Motor(2, "sts3095", MotorNormMode.DEGREES),
        "joint_3": Motor(3, "sts3250", MotorNormMode.DEGREES),
        "joint_4": Motor(4, "sts3095", MotorNormMode.DEGREES),
        "joint_5": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "joint_6": Motor(6, "sts3215", MotorNormMode.DEGREES),
        "joint_7": Motor(7, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
    }
    odrive_motors = {"joint_1": Motor(GEM_ODRIVE_AXIS, "gim6010-8", MotorNormMode.DEGREES)}

    feetech_calibration = None
    odrive_calibration = None
    if not args.follower_raw:
        path = _follower_calibration_path(args.follower_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Follower calibration file not found at '{path}'. Run follower calibration first or pass --follower-raw."
            )
        all_cal = _load_calibration(path)
        feetech_calibration = {k: v for k, v in all_cal.items() if k in feetech_motors}
        odrive_calibration = {k: v for k, v in all_cal.items() if k in odrive_motors}

    feetech = FeetechMotorsBus(
        port=args.follower_feetech_port,
        motors=feetech_motors,
        calibration=feetech_calibration,
    )
    odrive = ODriveMotorsBus(
        port=args.follower_odrive_port,
        motors=odrive_motors,
        calibration=odrive_calibration,
    )

    feetech.connect()
    odrive.connect()
    feetech.disable_torque()
    odrive.disable_torque()
    return feetech, odrive


def _read_follower_positions(
    feetech: FeetechMotorsBus, odrive: ODriveMotorsBus, raw: bool
) -> dict[str, float]:
    positions = feetech.sync_read("Present_Position", normalize=not raw)
    positions["joint_1"] = float(odrive.read("Present_Position", "joint_1"))
    return positions


def render_leader(action: dict[str, float]) -> None:
    print("Minion Leader")
    print("-------------")
    for name in JOINT_ORDER:
        value = action.get(f"{name}.pos", float("nan"))
        print(f"{name:8s}  {value:8.2f}")


def render_follower(positions: dict[str, float]) -> None:
    print("GEM Follower (read-only, torque disabled)")
    print("-----------------------------------------")
    for name in JOINT_ORDER:
        value = positions.get(name, float("nan"))
        print(f"{name:8s}  {value:8.2f}")


def render_both(leader_action: dict[str, float], follower_positions: dict[str, float]) -> None:
    print("Leader vs Follower")
    print("------------------")
    print(f"{'joint':8s}  {'leader':>10s}  {'follower':>10s}")
    for name in JOINT_ORDER:
        lv = leader_action.get(f"{name}.pos", float("nan"))
        fv = follower_positions.get(name, float("nan"))
        print(f"{name:8s}  {lv:10.2f}  {fv:10.2f}")


def render_frame(
    mode: str,
    leader_action: dict[str, float] | None,
    follower_positions: dict[str, float] | None,
    clear: bool = True,
) -> None:
    if clear:
        print("\x1b[2J\x1b[H", end="")

    if mode == "leader":
        render_leader(leader_action or {})
    elif mode == "follower":
        render_follower(follower_positions or {})
    else:
        render_both(leader_action or {}, follower_positions or {})


def main() -> None:
    args = parse_args()

    leader = None
    feetech = None
    odrive = None

    if args.mode in ("leader", "both"):
        leader = MinionArm(
            MinionArmConfig(
                id=args.leader_id,
                port=args.leader_port,
            )
        )
        leader.connect(calibrate=args.leader_calibrate)

    if args.mode in ("follower", "both"):
        feetech, odrive = _connect_follower(args)

    try:
        while True:
            leader_action = leader.get_action() if leader is not None else None
            follower_positions = (
                _read_follower_positions(feetech, odrive, args.follower_raw)
                if feetech is not None and odrive is not None
                else None
            )
            render_frame(
                mode=args.mode,
                leader_action=leader_action,
                follower_positions=follower_positions,
                clear=not args.no_clear,
            )
            time.sleep(max(1.0 / args.fps, 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        if leader is not None and leader.is_connected:
            leader.disconnect()
        if odrive is not None and odrive.is_connected:
            odrive.disconnect(disable_torque=True)
        if feetech is not None and feetech.is_connected:
            feetech.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
