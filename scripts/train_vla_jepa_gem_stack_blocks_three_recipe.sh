#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

DATASET_REPO="${DATASET_REPO:-jclinton1/gem_stack_blocks_three_20260615_114905}"
POLICY_PATH="${POLICY_PATH:-lerobot/VLA-JEPA-Pretrain}"
POLICY_REPO_ID="${POLICY_REPO_ID:-jclinton1/vla_jepa_gem_stack_blocks_three_20260615_114905}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-vla_jepa_gem_stack_blocks_three_pretrain_30k_bs1_paged_bnb_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}.log}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-20}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-6}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-5000}"

mkdir -p "${LOG_DIR}"

exec uv run --with bitsandbytes lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.device=cuda \
  --policy.torch_dtype=bfloat16 \
  --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]' \
  --policy.gripper_dim=7 \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO}" \
  --rename_map='{"observation.images.head":"observation.images.exterior_1_left","observation.images.wrist":"observation.images.exterior_2_left"}' \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --num_workers="${NUM_WORKERS}" \
  --prefetch_factor=4 \
  --log_freq="${LOG_FREQ}" \
  --eval_freq="${EVAL_FREQ}" \
  --validation_split=0.1 \
  --validation_samples="${VALIDATION_SAMPLES:-256}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable="${WANDB_ENABLE:-true}" \
  --wandb.project="${WANDB_PROJECT:-gem-stack-blocks}" \
  --wandb.group="${WANDB_GROUP:-gem-stack-blocks-three-vla-jepa}" \
  --seed=1000 \
  --job_name="${JOB_NAME}" \
  --output_dir="${OUTPUT_DIR}" \
  --optimizer.type=bnb-adamw8bit \
  --optimizer.lr="${OPTIMIZER_LR}" \
  --optimizer.betas='[0.9, 0.95]' \
  --optimizer.eps=1e-8 \
  --optimizer.weight_decay=1e-8 \
  --optimizer.grad_clip_norm=1.0 \
  --optimizer.paged=true \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.peak_lr="${OPTIMIZER_LR}" \
  --scheduler.decay_lr="${SCHEDULER_DECAY_LR}" \
  --scheduler.num_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  --scheduler.num_decay_steps="${STEPS}" \
  --use_policy_training_preset=false \
  2>&1 | tee "${LOG_FILE}"
