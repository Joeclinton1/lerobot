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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any  # Added Optional

import torch
import torch.profiler
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser

# Assume TrainPipelineConfig has validation_dataset and validation_freq attributes
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type, enabled=use_amp):
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    # pack both scalars into one tiny tensor
    stats_gpu = torch.stack([loss.detach(), grad_norm.detach()])

    # 1 sync to host
    stats_cpu = stats_gpu.to("cpu", non_blocking=True)
    loss_val, grad_norm_val = stats_cpu.tolist()

    train_metrics.loss = loss_val
    train_metrics.grad_norm = grad_norm_val
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    # convert outputs as before, or defer them too
    loggable_output_dict = {}
    for k, v in (output_dict or {}).items():
        if isinstance(v, torch.Tensor):
            cpu_v = v.detach().cpu()  # single sync per output tensor
            loggable_output_dict[k] = cpu_v.item() if cpu_v.numel() == 1 else cpu_v.numpy()
        else:
            loggable_output_dict[k] = v

    return train_metrics, loggable_output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    policy = policy.to(device)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Episode-level split
    num_eps = len(dataset.episode_data_index["from"])
    val_size = int(num_eps * 0.1)
    g = torch.Generator().manual_seed(cfg.seed or 42)
    shuffled = torch.randperm(num_eps, generator=g).tolist()
    split_episodes = {"train": shuffled[:-val_size], "val": shuffled[-val_size:]}

    # Build datasets and dataloaders
    dataloaders = {}
    for split, eps in split_episodes.items():
        indices = list(
            EpisodeAwareSampler(
                dataset.episode_data_index,
                episode_indices_to_use=eps,
                drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                shuffle=True,
            )
        )
        subset = torch.utils.data.Subset(dataset, indices)
        dataloaders[split] = torch.utils.data.DataLoader(
            subset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=(split == "train"),
        )

    dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "val_loss": AverageMeter("val_loss", ":.4f"),
        "val_s": AverageMeter("val_s", ":.4f"),  # val_s metric for amortized cost of validation per step
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    # Use current_step for iteration, step for external tracking
    for current_step in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step = current_step + 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = cfg.save_freq > 0 and (step % cfg.save_freq == 0 or step == cfg.steps)
        is_validation_step = (
            val_dataloader is not None and cfg.validation_freq > 0 and step % cfg.validation_freq == 0
        )
        is_eval_step = eval_env is not None and cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # Integrate validation loop directly here
        if is_validation_step:
            validation_start_time = time.perf_counter()

            policy.eval()
            total_val_loss = 0.0
            fraction = 0.05  # only sample 8% of val dataset. Hardcoded for simplicity.
            num_batches_to_run = int(fraction * len(val_dataloader))
            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_dataloader):
                    if batch_idx >= num_batches_to_run:
                        break
                    for key in val_batch:
                        if isinstance(val_batch[key], torch.Tensor):
                            val_batch[key] = val_batch[key].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=cfg.policy.use_amp):
                        loss, _ = policy.forward(val_batch)
                    if torch.isfinite(loss):
                        total_val_loss += loss.item()

            policy.train()
            avg_val_loss = total_val_loss / num_batches_to_run
            train_tracker.metrics["val_loss"].reset()
            train_tracker.metrics["val_loss"].update(avg_val_loss)

            validation_duration = time.perf_counter() - validation_start_time
            amortized_val_time = validation_duration / cfg.validation_freq

            train_tracker.metrics["val_s"].reset()
            train_tracker.metrics["val_s"].update(amortized_val_time)

        if is_log_step:
            log_message = str(train_tracker)  # Get base log string
            logging.info(log_message)

            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()

                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log main metrics without prefix for minimal change, relies on user knowing context
                wandb_logger.log_dict(wandb_log_dict, step=step)

            # Reset only training-related averages
            train_tracker.reset_averages(keys=["loss", "grad_norm", "lr", "update_s", "dataloading_s"])

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=cfg.policy.use_amp):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed + step if cfg.seed is not None else None,
                )

            # Minimal console log for eval
            logging.info(f"Eval results step {step}: {eval_info['aggregated']}")

            if wandb_logger:
                # Log aggregated eval results without prefix for minimal change
                wandb_logger.log_dict(eval_info["aggregated"], step=step)
                if eval_info.get("video_paths"):
                    try:
                        wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
                    except Exception as e:
                        logging.warning(f"Failed to log evaluation video to WandB: {e}")

    # --- End of Training ---
    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
