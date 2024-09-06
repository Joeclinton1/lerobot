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
"""Process zarr files formatted for BlockPush dataset (like in https://github.com/real-stanford/diffusion_polic) to LeRobot format"""

import numpy as np
import torch
import zarr
from pathlib import Path
from datasets import Dataset, Features, Sequence, Value

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import calculate_episode_data_index, hf_transform_to_torch

def check_format(raw_dir):
    zarr_path = raw_dir / "blockpush_multimodal.zarr"
    zarr_data = zarr.open(zarr_path, mode="r")

    required_datasets = {"data/obs", "data/action", "meta/episode_ends"}
    for dataset in required_datasets:
        assert dataset in zarr_data, f"Required dataset {dataset} not found in zarr file"
    nb_frames = zarr_data["data/obs"].shape[0]

    required_datasets.remove("meta/episode_ends")

    assert all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets), \
        "Mismatch in number of frames across datasets"

def load_from_raw(
    raw_dir: Path,
    fps: int,
    episodes: list[int] | None = None
):
    zarr_path = raw_dir / "blockpush_multimodal.zarr"
    zarr_data = zarr.open(zarr_path, mode="r")

    observations = torch.from_numpy(zarr_data["data/obs"][:])
    actions = torch.from_numpy(zarr_data["data/action"][:])
    episode_ends = zarr_data["meta/episode_ends"][:]

    from_ids = [0] + episode_ends[:-1].tolist()
    to_ids = episode_ends.tolist()

    num_episodes = len(from_ids)

    ep_dicts = []
    ep_ids = episodes if episodes is not None else range(num_episodes)
    for ep_idx, selected_ep_idx in enumerate(ep_ids):
        from_idx = from_ids[selected_ep_idx]
        to_idx = to_ids[selected_ep_idx]
        num_frames = to_idx - from_idx
        obs = observations[from_idx:to_idx]
        ep_dict = {
            "observation.state":obs[:, 6:10], # effector translation and effector target translation
            "observation.environment_state": torch.cat((obs[:, :6], obs[:, 10:]), dim=-1), # other observation values
            "action": actions[from_idx:to_idx],
            "episode_index": torch.full((num_frames,), ep_idx, dtype=torch.int64),
            "frame_index": torch.arange(num_frames, dtype=torch.int64),
            "timestamp": torch.arange(num_frames, dtype=torch.float32) / fps,
            "next.reward": torch.zeros(num_frames, dtype=torch.float32),  # NOTE: Placeholder
            "next.done": torch.zeros(num_frames, dtype=torch.bool),
            "next.success": torch.zeros(num_frames, dtype=torch.bool),  # NOTE: Placeholder
        }
        ep_dict["next.done"][-1] = True

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)
    data_dict["index"] = torch.arange(len(data_dict["frame_index"]), dtype=torch.int64)
    return data_dict

def to_hf_dataset(data_dict):
    features = {
        "observation.state": Sequence(
            length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32")
        ),
        "observation.environment_state": Sequence(
            length=data_dict["observation.environment_state"].shape[1], feature=Value(dtype="float32")
        ),
        "action": Sequence(
            length=data_dict["action"].shape[1], feature=Value(dtype="float32")
        ),
        "episode_index": Value(dtype="int64"),
        "frame_index": Value(dtype="int64"),
        "timestamp": Value(dtype="float32"),
        "next.reward": Value(dtype="float32"),
        "next.done": Value(dtype="bool"),
        "next.success": Value(dtype="bool"),
        "index": Value(dtype="int64")
    }

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    check_format(raw_dir)

    if fps is None:
        fps = 10

    data_dict = load_from_raw(raw_dir, fps, episodes)
    hf_dataset = to_hf_dataset(data_dict)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": False,
    }

    return hf_dataset, episode_data_index, info