import logging
import shutil
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from lerobot.types import RobotAction

logger = logging.getLogger(__name__)

GEM_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")
SO_TO_GEM = {
    "shoulder_pan": "joint_1",
    "shoulder_lift": "joint_2",
    "elbow_flex": "joint_3",
    "wrist_flex": "joint_4",
    "wrist_roll": "joint_5",
    "gripper": "gripper",
}


@dataclass
class RobotArmViewerConfig:
    enabled: bool = False
    only: bool = False
    viewer_url: str = "http://127.0.0.1:8020"
    launch_viewer: bool = True
    viewer_host: str = "127.0.0.1"
    viewer_port: int = 8020
    viewer_root: Path | None = None
    arm_spacing_m: float = 0.2
    open_browser: bool = True


def map_action_for_viewer(action: RobotAction, robot_type: str) -> RobotAction:
    if _viewer_model_name(robot_type) == "GEM":
        return _map_action_to_gem(action)
    return {key: float(value) for key, value in action.items()}


def map_action_to_gem(action: RobotAction) -> RobotAction:
    return _map_action_to_gem(action)


def _map_action_to_gem(action: RobotAction) -> RobotAction:
    mapped: RobotAction = {}
    for key, value in action.items():
        name = key.removesuffix(".pos")
        side, joint = _split_side(name)
        gem_joint = SO_TO_GEM.get(joint, joint)
        if gem_joint not in GEM_JOINTS:
            continue
        prefix = f"{side}_" if side is not None else ""
        mapped[f"{prefix}{gem_joint}.pos"] = float(value)
    return mapped


class RobotArmViewer:
    """Sidecar that mirrors single GEM joint actions into robot-arm-viewer."""

    def __init__(self, config: RobotArmViewerConfig, robot_type: str):
        self.config = config
        self.robot_type = robot_type
        self.model_name = _viewer_model_name(robot_type)
        self._process: subprocess.Popen | None = None
        self._connected = False
        self._mode = "single"

    def connect(self) -> None:
        if self.config.launch_viewer:
            self._start_viewer()
        self._wait_until_ready()
        self._post(
            "configure",
            {
                "mode": self._mode,
                "robot": self.model_name,
                "spacing_m": self.config.arm_spacing_m,
                "leader_control": True,
                "load_sidecar": self.model_name == "GEM",
                "fast_sidecar": True,
            },
        )
        if self.config.open_browser:
            webbrowser.open(self.config.viewer_url)
        self._connected = True
        logger.info("Robot arm viewer connected to %s", self.config.viewer_url)

    def send_action(self, action: RobotAction) -> None:
        if not self._connected:
            return
        if _is_bimanual_action(action) and self._mode != "dual":
            self._mode = "dual"
            self._post(
                "configure",
                {
                    "mode": self._mode,
                    "robot": self.model_name,
                    "spacing_m": self.config.arm_spacing_m,
                    "leader_control": True,
                    "load_sidecar": self.model_name == "GEM",
                    "fast_sidecar": True,
                },
            )
        self._post("action", {"actions": map_action_for_viewer(action, self.robot_type)})

    def disconnect(self) -> None:
        self._connected = False
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def _start_viewer(self) -> None:
        if self._viewer_is_ready():
            return
        self._process = subprocess.Popen(self._viewer_command())  # noqa: S603, S607

    def _viewer_command(self) -> list[str]:
        if self.config.viewer_root is not None:
            script = Path(self.config.viewer_root) / "bin" / "robot-arm-viewer.js"
            return ["node", str(script), "--host", self.config.viewer_host, "--port", str(self.config.viewer_port)]
        bundled_script = Path("C:/github_personal/urdf-loaders-obj/bin/robot-arm-viewer.js")
        if bundled_script.is_file():
            return [
                "node",
                str(bundled_script),
                "--host",
                self.config.viewer_host,
                "--port",
                str(self.config.viewer_port),
            ]
        if shutil.which("robot-arm-viewer"):
            return ["robot-arm-viewer", "--host", self.config.viewer_host, "--port", str(self.config.viewer_port)]
        return ["robot-arm-viewer", "--host", self.config.viewer_host, "--port", str(self.config.viewer_port)]

    def _wait_until_ready(self) -> None:
        deadline = time.perf_counter() + 10
        while time.perf_counter() < deadline:
            if self._viewer_is_ready():
                return
            time.sleep(0.1)
        raise RuntimeError(f"robot-arm-viewer did not become ready at {self.config.viewer_url}")

    def _viewer_is_ready(self) -> bool:
        try:
            response = requests.get(f"{self.config.viewer_url}/health", timeout=0.3)
            return response.ok
        except requests.RequestException:
            return False

    def _post(self, endpoint: str, payload: dict[str, Any]) -> None:
        response = requests.post(f"{self.config.viewer_url}/api/{endpoint}", json=payload, timeout=0.5)
        response.raise_for_status()


def _split_side(name: str) -> tuple[str | None, str]:
    if name.startswith("left_"):
        return "left", name.removeprefix("left_")
    if name.startswith("right_"):
        return "right", name.removeprefix("right_")
    return (None, name)


def _viewer_model_name(robot_type: str) -> str:
    if robot_type in {"so100_follower", "so101_follower"}:
        return "SO-ARM101"
    return "GEM"


def _is_bimanual_action(action: RobotAction) -> bool:
    return any(key.startswith(("left_", "right_")) for key in action)
