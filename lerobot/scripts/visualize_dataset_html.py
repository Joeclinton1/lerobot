#!/usr/bin/env python

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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import shutil
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, url_for
from scipy.spatial.transform import Rotation

from lerobot import available_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import IterableNamespace
from lerobot.common.utils.utils import init_logging

###############################################################################
# Helper functions for geometry rendering                                   #
###############################################################################


# ruff: noqa: N806
def vec_to_transform(vec: np.ndarray) -> np.ndarray:
    """Convert a 7‑element vec (3 pos + 4 quat) into a 4×4 homogeneous matrix."""
    pos = vec[:3]
    euler = vec[3:6]
    R = Rotation.from_euler("xyz", euler).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


###############################################################################
# NEW – lightweight EE‑space payload for the browser                         #
###############################################################################


def _make_line_trace(
    x: list[float], y: list[float], z: list[float], *, colour: str, name: str
) -> dict[str, Any]:
    """Return a Plotly 3‑D scatter line trace as a plain dictionary."""
    return {
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "line": {"color": colour},
        "name": name,
    }


def _make_marker_trace(x: float, y: float, z: float, *, colour: str, name: str) -> dict[str, Any]:
    """Return a single‑point marker trace as a plain dictionary."""
    return {
        "type": "scatter3d",
        "mode": "markers",
        "x": [x],
        "y": [y],
        "z": [z],
        "marker": {"size": 6, "color": colour, "symbol": "square"},
        "showlegend": False,
        "name": name,
    }


def generate_interactive_ee_plot(obs_space, act_space):
    import numpy as np
    from scipy.spatial.transform import Rotation

    # ──────── parameters ──────────────────────────────────────────────
    arrow_length = 0.085  # meters
    finger_offset = 0.04  # meters

    # ──────── helpers ─────────────────────────────────────────────────
    def to_py(arr):
        return [[float(v) for v in row] for row in arr]

    def make_line(start, end, color):
        return {
            "type": "scatter3d",
            "mode": "lines",
            "x": [start[0], end[0]],
            "y": [start[1], end[1]],
            "z": [start[2], end[2]],
            "line": {"width": 5, "color": color},
            "showlegend": False,
        }

    def pose_arrows(pose, grip):
        # compute base and finger arrows for one frame
        grip = grip * math.pi / 180  # if needed
        T = vec_to_transform(pose)
        pos = T[:3, 3]
        forward = T[:3, 0]

        # project global Z onto plane ⟂ forward → offset direction
        world_up = np.array([0.0, 0.0, 1.0])
        proj = world_up - forward * (world_up @ forward)
        offset_dir = proj / np.linalg.norm(proj)

        # base arrow tip
        base_tip = pos + arrow_length * forward

        # finger start & tip
        finger_start = pos + finger_offset * offset_dir
        hinge_axis = np.cross(forward, offset_dir)
        hinge_axis /= np.linalg.norm(hinge_axis)
        finger_dir = Rotation.from_rotvec(grip * hinge_axis).apply(forward)
        finger_tip = finger_start + arrow_length * finger_dir

        return base_tip.tolist(), [finger_start.tolist(), finger_tip.tolist()]

    # ──────── prepare trajectories & markers ──────────────────────────────
    obs_pts = to_py(obs_space[:, :3])
    act_pts = to_py(act_space[:, :3])
    grips = [float(g) for g in act_space[:, -1]]

    data = [
        _make_line_trace(*zip(*obs_pts, strict=False), colour="red", name="Obs Trajectory"),
        _make_marker_trace(*obs_pts[0], colour="red", name="Obs EE Marker"),
        _make_line_trace(*zip(*act_pts, strict=False), colour="green", name="Act Trajectory"),
        _make_marker_trace(*act_pts[0], colour="green", name="Act EE Marker"),
    ]

    # ──────── compute arrows for each frame ────────────────────────────────
    obs_bases, obs_fingers = zip(
        *(pose_arrows(p, g) for p, g in zip(obs_space, grips, strict=False)), strict=False
    )
    act_bases, act_fingers = zip(
        *(pose_arrows(p, g) for p, g in zip(act_space, grips, strict=False)), strict=False
    )
    obs_bases, act_bases = list(obs_bases), list(act_bases)
    obs_fingers, act_fingers = list(obs_fingers), list(act_fingers)

    # ──────── initial arrows (frame 0) ─────────────────────────────────────
    data += [
        # base arrows
        make_line(obs_pts[0], obs_bases[0], "red"),
        make_line(act_pts[0], act_bases[0], "green"),
        # finger arrows (light colours)
        make_line(*obs_fingers[0], "#ff8080"),
        make_line(*act_fingers[0], "#80ff80"),
        # **new** L‑connector: origin → finger_start
        make_line(obs_pts[0], obs_fingers[0][0], "#ff8080"),
        make_line(act_pts[0], act_fingers[0][0], "#80ff80"),
    ]

    # ──────── layout & return ──────────────────────────────────────────────
    layout = {
        "scene": {
            "xaxis": {
                "title": "X",
                "dtick": 0.05,
            },
            "yaxis": {"title": "Y", "dtick": 0.05},
            "zaxis": {"title": "Z", "dtick": 0.05},
            "aspectmode": "cube",
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
    }

    return {
        "data": data,
        "layout": layout,
        "marker_coords": {"obs": obs_pts, "act": act_pts},
        "arrow_coords": {
            "obs_base": obs_bases,
            "act_base": act_bases,
            "obs_finger": obs_fingers,
            "act_finger": act_fingers,
            # newly added connector points
            "obs_connector": [[obs_pts[i], obs_fingers[i][0]] for i in range(len(obs_pts))],
            "act_connector": [[act_pts[i], act_fingers[i][0]] for i in range(len(act_pts))],
        },
    }


###############################################################################
# Un‑modified helpers for fetching dataset data                              #
###############################################################################
def get_episode_ee_space(
    dataset: LeRobotDataset | IterableNamespace,
    episode_index: int,
    mode: str = "obs",  # "obs" for observation.state, "act" for action
) -> np.ndarray:
    """Extract EE pose (pos + quat) for obs or act."""
    field = "observation.state" if mode == "obs" else "action"
    if isinstance(dataset, LeRobotDataset):
        frm, to = (
            dataset.episode_data_index["from"][episode_index],
            dataset.episode_data_index["to"][episode_index],
        )
        df = dataset.hf_dataset.select(range(frm, to)).with_format("pandas")
    else:
        url = f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/main/" + dataset.data_path.format(
            episode_chunk=episode_index // dataset.chunks_size, episode_index=episode_index
        )
        df = pd.read_parquet(url)
    arr = np.vstack(df[field])
    return arr[:, :7] if arr.shape[1] > 7 else arr


###############################################################################
# … the remainder of the original script is unchanged apart from             #
# a very small tweak in ``show_episode``: no extra JSON conversion is needed  #
# – we pass the pure‑Python dict returned above directly to Jinja.            #
###############################################################################

# (Everything below here is identical to the original file except that
# comments pointing to the old EE‑space logic were removed for brevity.)


def get_episode_data(dataset: LeRobotDataset | IterableNamespace, episode_index: int):
    """Generate a CSV string with timeseries data (state, action, etc.) for an episode."""
    columns = []
    selected_columns = [col for col, ft in dataset.features.items() if ft["dtype"] in ["float32", "int32"]]
    selected_columns.remove("timestamp")
    ignored_columns: list[str] = []
    for column_name in selected_columns.copy():
        shape = dataset.features[column_name]["shape"]
        if len(shape) > 1:
            selected_columns.remove(column_name)
            ignored_columns.append(column_name)
    header = ["timestamp"]
    for column_name in selected_columns:
        if isinstance(dataset, LeRobotDataset):
            dim_state = dataset.meta.shapes[column_name][0]
        else:
            dim_state = dataset.features[column_name].get("shape", [])[0]
        if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
            column_names = dataset.features[column_name]["names"]
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})
        header += column_names
    selected_columns.insert(0, "timestamp")
    if isinstance(dataset, LeRobotDataset):
        from_idx = dataset.episode_data_index["from"][episode_index]
        to_idx = dataset.episode_data_index["to"][episode_index]
        data = (
            dataset.hf_dataset.select(range(from_idx, to_idx))
            .select_columns(selected_columns)
            .with_format("pandas")
        )
    else:
        repo_id = dataset.repo_id
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/" + dataset.data_path.format(
            episode_chunk=int(episode_index) // dataset.chunks_size, episode_index=episode_index
        )
        data = pd.read_parquet(url)[selected_columns]
    rows = np.hstack(
        (np.expand_dims(data["timestamp"], 1), *[np.vstack(data[col]) for col in selected_columns[1:]])
    ).tolist()
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(header)
    csv_writer.writerows(rows)
    return csv_buffer.getvalue(), columns, ignored_columns


def get_dataset_info(repo_id: str) -> IterableNamespace:
    """Retrieve dataset information from the Hugging Face hub."""
    response = requests.get(
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json", timeout=5
    )
    response.raise_for_status()
    dataset_info = response.json()
    dataset_info["repo_id"] = repo_id
    return IterableNamespace(dataset_info)


def run_server(
    dataset: LeRobotDataset | IterableNamespace | None,
    episodes: list[int] | None,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
    use_ee_space: bool = False,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.route("/")
    def homepage(dataset=dataset):
        if dataset:
            dataset_namespace, dataset_name = dataset.repo_id.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )
        dataset_param = request.args.get("dataset")
        episode_param = request.args.get("episode", 0, type=int)
        if dataset_param:
            dataset_namespace, dataset_name = dataset_param.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=episode_param,
                )
            )
        featured_datasets = [
            "lerobot/aloha_static_cups_open",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/taco_play",
        ]
        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=featured_datasets,
            lerobot_datasets=available_datasets,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>")
    def show_first_episode(dataset_namespace, dataset_name):
        return redirect(
            url_for(
                "show_episode", dataset_namespace=dataset_namespace, dataset_name=dataset_name, episode_id=0
            )
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id, dataset=dataset, episodes=episodes):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        try:
            if dataset is None:
                dataset = get_dataset_info(repo_id)
        except FileNotFoundError:
            return (
                "Make sure to convert your LeRobotDataset to v2 & above. "
                "See https://github.com/huggingface/lerobot/pull/461"
            ), 400
        dataset_version = (
            str(dataset.meta._version) if isinstance(dataset, LeRobotDataset) else dataset.codebase_version
        )
        match = re.search(r"v(\d+)\.", dataset_version)
        if match and int(match.group(1)) < 2:
            return "Make sure to convert your LeRobotDataset to v2 & above."

        episode_data_csv_str, columns, ignored_columns = get_episode_data(dataset, episode_id)
        dataset_info = {
            "repo_id": repo_id,
            "num_samples": dataset.num_frames
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_frames,
            "num_episodes": dataset.num_episodes
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_episodes,
            "fps": dataset.fps,
        }
        if isinstance(dataset, LeRobotDataset):
            video_paths = [
                dataset.meta.get_video_file_path(episode_id, key) for key in dataset.meta.video_keys
            ]
            videos_info = [
                {"url": url_for("static", filename=video_path), "filename": video_path.parent.name}
                for video_path in video_paths
            ]
            tasks = dataset.meta.episodes[episode_id]["tasks"]
        else:
            video_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "video"]
            videos_info = [
                {
                    "url": (
                        f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
                        + dataset.video_path.format(
                            episode_chunk=int(episode_id) // dataset.chunks_size,
                            video_key=video_key,
                            episode_index=episode_id,
                        )
                    ),
                    "filename": video_key,
                }
                for video_key in video_keys
            ]
            response = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/episodes.jsonl", timeout=5
            )
            response.raise_for_status()
            tasks_jsonl = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            tasks = next(row["tasks"] for row in tasks_jsonl if row["episode_index"] == episode_id)
        videos_info[0]["language_instruction"] = tasks
        if episodes is None:
            episodes = list(
                range(dataset.num_episodes if isinstance(dataset, LeRobotDataset) else dataset.total_episodes)
            )

        # ───── EE‑space data ─────
        if use_ee_space:
            try:
                obs_space = get_episode_ee_space(dataset, episode_id, mode="obs")
                act_space = get_episode_ee_space(dataset, episode_id, mode="act")
                ee_space_plot_data: dict[str, Any] | None = generate_interactive_ee_plot(obs_space, act_space)
            except Exception as exc:
                app.logger.error("Error preparing interactive EE‑space data: %s", exc)
                ee_space_plot_data = None
        else:
            ee_space_plot_data = None

        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            episode_data_csv_str=episode_data_csv_str,
            columns=columns,
            ignored_columns=ignored_columns,
            ee_space_plot_data=ee_space_plot_data,
        )

    app.run(host=host, port=port)


###############################################################################
# CLI entry‑point                                                            #
###############################################################################


def visualize_dataset_html(
    dataset: LeRobotDataset | None,
    episodes: list[int] | None = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
    use_ee_space: bool = False,
) -> Path | None:
    init_logging()
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")
    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    if dataset is not None and isinstance(dataset, LeRobotDataset):
        ln_videos_dir = static_dir / "videos"
        if not ln_videos_dir.exists():
            ln_videos_dir.symlink_to((dataset.root / "videos").resolve())
    if serve:
        run_server(dataset, episodes, host, port, static_dir, template_dir, use_ee_space)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default=None, help="HF repository id (e.g. 'lerobot/pusht').")
    parser.add_argument("--root", type=Path, default=None, help="Root directory for a local dataset.")
    parser.add_argument(
        "--load-from-hf-hub", type=int, default=0, help="Load videos & parquet from the HF Hub."
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices to visualise.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for HTML & static assets.")
    parser.add_argument("--serve", type=int, default=1, help="Launch the webserver.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Webserver host.")
    parser.add_argument("--port", type=int, default=9090, help="Webserver port.")
    parser.add_argument("--force-override", type=int, default=0, help="Delete output directory if exists.")
    parser.add_argument("--tolerance-s", type=float, default=1e-4, help="Tolerance for fps value.")
    parser.add_argument("--use-ee-space", action="store_true", help="Include interactive EE trajectories.")
    args = parser.parse_args()

    repo_id = args.repo_id
    load_from_hf_hub = bool(args.load_from_hf_hub)
    root = args.root
    tolerance_s = args.tolerance_s

    dataset: LeRobotDataset | IterableNamespace | None = None
    if repo_id:
        dataset = (
            LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
            if not load_from_hf_hub
            else get_dataset_info(repo_id)
        )

    visualize_dataset_html(
        dataset,
        episodes=args.episodes,
        output_dir=args.output_dir,
        serve=bool(args.serve),
        host=args.host,
        port=args.port,
        force_override=bool(args.force_override),
        use_ee_space=args.use_ee_space,
    )


if __name__ == "__main__":
    main()
