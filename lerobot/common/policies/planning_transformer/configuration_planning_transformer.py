#!/usr/bin/env python

# Copyright 2024 Your Name or Organization
# All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os
import re
import uuid


@dataclass
class TrainConfig:
    """Configuration class for training.

    This configuration supports standard reinforcement learning tasks and ablation testing.

    Args:
        project: WandB project name.
        group: WandB group name.
        name: WandB run name.
        embedding_dim: Transformer hidden dimension.
        num_layers: Depth of the transformer model.
        num_heads: Number of heads in the attention mechanism.
        seq_len: Maximum sequence length during training.
        episode_len: Maximum rollout length for positional embeddings.
        attention_dropout: Attention dropout probability.
        residual_dropout: Residual dropout probability.
        embedding_dropout: Embedding dropout probability.
        max_action: Maximum range for symmetric actions, typically [-1, 1].
        env_name: Name of the training dataset and evaluation environment.
        learning_rate: Learning rate for AdamW optimizer.
        betas: Beta coefficients for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        clip_grad: Maximum gradient norm during training (optional).
        batch_size: Batch size for training.
        update_steps: Total number of training steps.
        warmup_steps: Number of warmup steps for the learning rate scheduler.
        reward_scale: Reward scaling factor to reduce magnitude.
        num_workers: Number of workers for PyTorch DataLoader.
        target_returns: Target return-to-go values for evaluation.
        eval_episodes: Number of episodes to evaluate.
        eval_every: Evaluation frequency in training steps.
        checkpoints_path: Path for saving checkpoints (optional).
        deterministic_torch: Use deterministic algorithms in PyTorch.
        train_seed: Random seed for training.
        eval_seed: Random seed for evaluation.
        device: Device to use for training ("cuda" or "cpu").
        log_attn_weights: Whether to log attention weights.
        log_attn_every: Frequency of attention weight logging.
        plan_sampling_method: Method for sampling plans (optional).
        demo_mode: Run in demo mode.
        checkpoint_to_load: Checkpoint path to load (optional).
        other_params: Additional configuration options for testing.
    """

    # WandB configuration
    project: str = "CORL"
    group: str = "DT-D4RL"
    name: str = "PDT"

    # Transformer configuration
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1

    # Environment configuration
    max_action: float = 1.0
    env_name: str = "halfcheetah-medium-v2"

    # Optimizer configuration
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25

    # Training configuration
    batch_size: int = 64
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4

    # Evaluation configuration
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_episodes: int = 100
    eval_every: int = 10_000

    # Checkpoint configuration
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"
    log_attn_weights: bool = False
    log_attn_every: int = 100

    # Planning parameters
    num_plan_points: int = 10
    replanning_interval: int = 40
    plan_sampling_method: Optional[int] = 4
    plan_use_relative_states: Optional[bool] = True
    is_goal_conditioned: Optional[bool] = False
    goal_indices: Optional[Tuple[int, ...]] = (0, 1)
    plan_disabled: Optional[bool] = False

    # Video recording
    record_video: bool = False
    demo_mode: Optional[bool] = False
    checkpoint_to_load: Optional[str] = None

    # Other configuration
    action_noise_scale: Optional[float] = 0.4
    state_dim: Optional[int] = None
    disable_return_targets: Optional[bool] = False

    def __post_init__(self):
        """Post-initialization tasks for setting up dynamic configurations."""
        if self.demo_mode and self.checkpoint_to_load is None:
            regex = re.compile(r'^pdt_checkpoint_step=(\d+)\.pt$')
            latest_time = 0
            latest_folder_name = None

            for dirpath, _, filenames in os.walk(self.checkpoints_path or ""):
                for filename in filenames:
                    if match := regex.match(filename):
                        filepath = os.path.join(dirpath, filename)
                        mod_time = os.path.getmtime(filepath)
                        if mod_time > latest_time:
                            latest_time = mod_time
                            latest_folder_name = dirpath.split('/')[-1]
                            self.checkpoint_step_to_load = int(match.group(1))
            self.checkpoint_to_load = latest_folder_name

        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoint_to_load is not None:
            self.name = self.checkpoint_to_load
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

        if self.is_goal_conditioned is False:
            self.goal_indices = ()

        if self.plan_sampling_method is not None and self.plan_sampling_method not in {1, 2, 3, 4}:
            raise ValueError("Invalid plan sampling method. Must be 1, 2, 3, or 4.")
