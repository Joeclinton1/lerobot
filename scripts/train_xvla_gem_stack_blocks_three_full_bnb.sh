#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

DATASET_REPO="${DATASET_REPO:-jclinton1/gem_stack_blocks_three_20260502_194746}"
POLICY_PATH="${POLICY_PATH:-outputs/pretrained/xvla-base-2cam}"
STEPS="${STEPS:-200000}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-xvla_gem_stack_blocks_three_2cam_full_bnb_val2k_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}.log}"

mkdir -p "${LOG_DIR}"

exec uv run --with bitsandbytes lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --policy.optimizer_use_bnb_8bit=true \
  --policy.optimizer_bnb_paged=true \
  --policy.optimizer_lr="${OPTIMIZER_LR:-1e-4}" \
  --policy.optimizer_betas='[0.9, 0.95]' \
  --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay=0.0001 \
  --policy.optimizer_grad_clip_norm=10.0 \
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS:-1000}" \
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS:-${STEPS}}" \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR:-2.5e-6}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO}" \
  --rename_map='{"observation.images.head":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE:-1}" \
  --eval.batch_size=1 \
  --num_workers="${NUM_WORKERS:-2}" \
  --prefetch_factor="${PREFETCH_FACTOR:-4}" \
  --log_freq="${LOG_FREQ:-50}" \
  --eval_freq="${EVAL_FREQ:-2000}" \
  --validation_split="${VALIDATION_SPLIT:-0.1}" \
  --validation_samples="${VALIDATION_SAMPLES:-256}" \
  --save_checkpoint="${SAVE_CHECKPOINT:-true}" \
  --save_freq="${SAVE_FREQ:-20000}" \
  --wandb.enable="${WANDB_ENABLE:-true}" \
  --wandb.project="${WANDB_PROJECT:-gem-stack-blocks}" \
  --wandb.group="${WANDB_GROUP:-gem-stack-blocks-three-xvla-rtc}" \
  --seed="${SEED:-1000}" \
  --job_name="${JOB_NAME}" \
  --output_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
