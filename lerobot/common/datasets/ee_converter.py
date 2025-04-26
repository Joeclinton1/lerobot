#!/usr/bin/env python
"""
Convert LeRobot dataset from joint-space to EE-space and push to a new repo by appending '_ee' to the original repo id.
Orientations are represented as Euler angles (roll, pitch, yaw) instead of quaternions.
The conversion is applied to the whole dataset in memory, then saved back to per-episode parquet files efficiently.
Original data remains untouched.
Usage:
    python convert_ee.py --repo-id=lerobot/pusht
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from datasets import Dataset as HfDataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.utils import write_info, write_episode_stats, EPISODES_STATS_PATH
from lerobot.common.robot_devices.robots.kinematics import RobotKinematics
from collections import defaultdict


def pose_to_vec(ee_pose: np.ndarray, gripper: float) -> np.ndarray:
    """
    Convert an EE pose matrix and gripper value to a vector using Euler angles.
    Returns [x, y, z, roll, pitch, yaw, gripper].
    """
    pos = ee_pose[:3, 3]
    euler = Rotation.from_matrix(ee_pose[:3, :3]).as_euler('xyz', degrees=False)
    return np.concatenate([pos, euler, [gripper]]).astype(np.float32)


def convert_to_episode_dict(samples: list[dict]) -> dict[str, np.ndarray]:
    """
    Turn a list of sample dicts into a dict of arrays for HuggingFace or Arrow.
    """
    first = samples[0]
    episode_dict = {}
    for key, val in first.items():
        if isinstance(val, np.ndarray):
            episode_dict[key] = np.stack([s[key] for s in samples])
        else:
            episode_dict[key] = np.array([s[key] for s in samples])
    return episode_dict


def group_by_episode(samples: list[dict]) -> dict[int, list[dict]]:
    """
    Group samples by their episode_index field.
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[int(sample["episode_index"])].append(sample)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Source repo-id on HF Hub")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # 1. Load original dataset and kinematics
    logging.info(f"Loading dataset '{args.repo_id}'")
    src = LeRobotDataset(args.repo_id, force_cache_sync=True)
    kin = RobotKinematics(src.meta.robot_type)

    # 2. Clone on-disk files to new EE repo
    dst_root = src.root.parent / f"{src.root.name}_ee"
    logging.info(f"Cloning dataset files from '{src.root}' to '{dst_root}'")
    shutil.copytree(src.root, dst_root, dirs_exist_ok=True)
    new_repo_id = args.repo_id + "_ee"
    dst = LeRobotDataset(new_repo_id, root=dst_root)

    # 3. Update metadata shapes/names for EE-space
    features = dst.meta.info["features"]
    ee_names = ["ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw", "ee_gripper"]
    for key in ("observation.state", "action"):
        features[key]["shape"] = (7,)
        features[key]["names"] = ee_names
    write_info(dst.meta.info, dst_root)
    logging.info("Updated metadata for EE-space conversion using Euler angles.")

    # 4. Convert via HF map to EE-space
    logging.info("Converting dataset to EE-space with Euler angles...")
    dst.hf_dataset.reset_format()
    hf_dataset = dst.hf_dataset.map(
        lambda s: {
            **s,
            "observation.state": pose_to_vec(
                kin.fk_gripper(np.array(s["observation.state"])),
                float(s["observation.state"][-1])
            ),
            "action": pose_to_vec(
                kin.fk_gripper(np.array(s["action"])),
                float(s["action"][-1])
            ),
        },
        num_proc=4,
        load_from_cache_file=False,
    )

    # 5. Group samples in Python and write per-episode files & stats
    logging.info("Loading converted samples into memory for grouping...")
    all_samples = hf_dataset.with_format("python")  # list of dict
    episode_groups = group_by_episode(all_samples)

    logging.info("Writing per-episode parquet and computing stats...")
    dst.meta.episodes_stats = {}
    for ep_idx, samples in episode_groups.items():
        ep_path = dst_root / dst.meta.get_data_file_path(ep_idx)
        ep_path.parent.mkdir(parents=True, exist_ok=True)

        # Build HF Dataset from dict-of-arrays
        ep_dict = convert_to_episode_dict(samples)
        ep_ds = HfDataset.from_dict(ep_dict)
        ep_ds.to_parquet(str(ep_path))
        logging.info(f"Saved episode {ep_idx}")

        # Compute per-episode stats
        dst.meta.episodes_stats[ep_idx] = compute_episode_stats(ep_dict, dst.meta.features)

    # Retain any non-overwritten stats from source
    for ep_idx, src_stats in src.meta.episodes_stats.items():
        dst_stats = dst.meta.episodes_stats.get(ep_idx, {})
        dst_stats.update({k: v for k, v in src_stats.items() if k not in dst_stats})

    # Aggregate and persist global and per-episode stats
    dst.meta.stats = aggregate_stats(list(dst.meta.episodes_stats.values()))
    write_info(dst.meta.info, dst_root)
    (dst_root / EPISODES_STATS_PATH).write_text("")
    for ep_idx, stats in dst.meta.episodes_stats.items():
        write_episode_stats(ep_idx, stats, dst_root)
    logging.info("Episode statistics updated successfully.")

    # 6. Push to HF Hub
    logging.info(f"Pushing converted dataset to HF Hub as '{new_repo_id}'...")
    dst.push_to_hub()
    logging.info(f"Done: pushed '{new_repo_id}' to the Hub.")


if __name__ == "__main__":
    main()
