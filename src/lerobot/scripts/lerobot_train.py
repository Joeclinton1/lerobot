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
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import logging
import math
import time
from contextlib import nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, LeRobotDatasetMetadata, make_dataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all


class RandomSubsetSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        indices: list[int] | range,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> None:
        self.indices = list(indices)
        self.num_samples = min(num_samples, len(self.indices))
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError("RandomSubsetSampler requires at least one sample.")

    def __iter__(self):
        permutation = torch.randperm(len(self.indices), generator=self.generator)[: self.num_samples]
        for idx in permutation:
            yield self.indices[int(idx)]

    def __len__(self) -> int:
        return self.num_samples


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:
            # Use per-sample loss for weighted training
            # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Weighted loss: each sample's contribution is scaled by its weight.
            # We divide by weight sum (not batch size) so that if some weights are zero,
            # the remaining samples contribute proportionally more, preserving gradient scale.
            # Weights are pre-normalized to sum to batch_size for stable training dynamics.
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

            # Log weighting statistics
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def _split_train_validation_episodes(cfg: TrainPipelineConfig) -> tuple[list[int] | None, list[int] | None]:
    if cfg.validation_split <= 0:
        return cfg.dataset.episodes, None

    if not isinstance(cfg.dataset.repo_id, str):
        raise NotImplementedError("validation_split is only supported for single datasets.")

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    episodes = cfg.dataset.episodes or list(range(ds_meta.total_episodes))
    if len(episodes) < 2:
        raise ValueError("validation_split requires at least two episodes.")

    num_val_episodes = max(1, math.floor(len(episodes) * cfg.validation_split))
    if num_val_episodes >= len(episodes):
        raise ValueError(
            f"validation_split={cfg.validation_split} leaves no training episodes "
            f"for {len(episodes)} available episodes."
        )

    return episodes[:-num_val_episodes], episodes[-num_val_episodes:]


def _make_dataset_with_episodes(cfg: TrainPipelineConfig, episodes: list[int] | None):
    original_episodes = cfg.dataset.episodes
    cfg.dataset.episodes = episodes
    try:
        return make_dataset(cfg)
    finally:
        cfg.dataset.episodes = original_episodes


def _convert_uint8_images(batch: dict[str, Any], camera_keys: list[str]) -> None:
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0


def _infer_batch_size(batch: dict[str, Any]) -> int:
    if "index" in batch and hasattr(batch["index"], "shape"):
        return batch["index"].shape[0]

    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return value.shape[0]

    raise ValueError("Unable to infer batch size for validation loss.")


def _accumulate_numeric_outputs(
    output_sums: dict[str, torch.Tensor],
    output_dict: dict | None,
    batch_size: torch.Tensor,
    device: torch.device,
) -> None:
    if not output_dict:
        return

    for key, value in output_dict.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach()
        elif isinstance(value, (int | float)):
            value = torch.tensor(float(value), device=device)
        else:
            continue

        output_sums[key] = output_sums.get(key, torch.zeros((), device=device)) + value * batch_size


def validate_policy_loss(
    policy: PreTrainedPolicy,
    val_dataloader: torch.utils.data.DataLoader,
    preprocessor,
    camera_keys: list[str],
    accelerator: "Accelerator",
) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    loss_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    inner_model = getattr(loss_policy, "model", None)
    inner_model_was_training = inner_model.training if inner_model is not None else None
    start_time = time.perf_counter()

    loss_sum = torch.zeros((), device=accelerator.device)
    count_sum = torch.zeros((), device=accelerator.device)
    output_sums: dict[str, torch.Tensor] = {}

    try:
        if inner_model is not None and getattr(loss_policy.config, "use_vae", False):
            # ACT gates its VAE loss path on the inner model's training flag. Keep child modules in
            # eval mode, but enable that branch so validation measures the same objective as training.
            inner_model.training = True

        with torch.no_grad():
            for batch in val_dataloader:
                batch_size = torch.tensor(float(_infer_batch_size(batch)), device=accelerator.device)
                _convert_uint8_images(batch, camera_keys)
                batch = preprocessor(batch)
                with accelerator.autocast():
                    loss, output_dict = loss_policy.forward(batch)

                loss_sum += loss.detach() * batch_size
                count_sum += batch_size
                _accumulate_numeric_outputs(output_sums, output_dict, batch_size, accelerator.device)
    finally:
        if inner_model is not None and inner_model_was_training is not None:
            inner_model.training = inner_model_was_training

    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    count_sum = accelerator.reduce(count_sum, reduction="sum").clamp_min(1)
    metrics = {
        "val_loss": (loss_sum / count_sum).item(),
        "val_s": time.perf_counter() - start_time,
    }

    for key, value_sum in output_sums.items():
        value_sum = accelerator.reduce(value_sum, reduction="sum")
        metrics[f"val_{key}"] = (value_sum / count_sum).item()

    if was_training:
        policy.train()

    return metrics


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator

    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        train_episodes, val_episodes = _split_train_validation_episodes(cfg)
        if val_episodes is not None:
            logging.info(
                "Using validation split: %d train episodes, %d validation episodes",
                len(train_episodes or []),
                len(val_episodes),
            )
        dataset = _make_dataset_with_episodes(cfg, train_episodes)
        val_dataset = _make_dataset_with_episodes(cfg, val_episodes) if val_episodes is not None else None

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        train_episodes, val_episodes = _split_train_validation_episodes(cfg)
        dataset = _make_dataset_with_episodes(cfg, train_episodes)
        val_dataset = _make_dataset_with_episodes(cfg, val_episodes) if val_episodes is not None else None

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        logging.info("Using PEFT! Wrapping model.")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    if (
        getattr(active_cfg, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms if "
            "the checkpoint processors do not define them. Building processors from current policy config."
        )
        processor_pretrained_path = None

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            **processor_kwargs,
            **postprocessor_kwargs,
        )
        if cfg.rename_map:
            for step in preprocessor.steps:
                if getattr(step, "_registry_name", None) == "rename_observations_processor":
                    step.rename_map = cfg.rename_map
                    break

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(active_cfg, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=active_cfg.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    val_dataloader = None
    if val_dataset is not None:
        if hasattr(active_cfg, "drop_n_last_frames"):
            val_sampler = EpisodeAwareSampler(
                val_dataset.meta.episodes["dataset_from_index"],
                val_dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=val_dataset.episodes,
                drop_n_last_frames=active_cfg.drop_n_last_frames,
                shuffle=False,
            )
        else:
            val_sampler = None

        if cfg.validation_samples > 0:
            validation_indices = list(val_sampler) if val_sampler is not None else range(len(val_dataset))
            val_generator = torch.Generator()
            if cfg.seed is not None:
                val_generator.manual_seed(cfg.seed + 1)
            val_sampler = RandomSubsetSampler(
                validation_indices,
                num_samples=cfg.validation_samples,
                generator=val_generator,
            )
            logging.info(
                "Using random validation subset: %d of %d frames per validation pass",
                len(val_sampler),
                len(validation_indices),
            )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    if val_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, val_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        _convert_uint8_images(batch, dataset.meta.camera_keys)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log sample weighting statistics if enabled
                if sample_weighter is not None:
                    weighter_stats = sample_weighter.get_stats()
                    wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if val_dataloader is not None and is_eval_step:
            val_metrics = validate_policy_loss(
                policy,
                val_dataloader,
                preprocessor,
                val_dataset.meta.camera_keys,
                accelerator=accelerator,
            )
            if is_main_process:
                logging.info(
                    "Validation loss at step %s: %s",
                    step,
                    " ".join(f"{key}:{value:.4f}" for key, value in val_metrics.items()),
                )
                if wandb_logger:
                    wandb_logger.log_dict(val_metrics, step, mode="eval")

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
