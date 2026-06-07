#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

DATASET_REPO="${DATASET_REPO:-jclinton1/gem_stack_blocks_three_20260502_194746}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/gem_stack_blocks_three_rel30_stats}"
POLICY_PATH="${POLICY_PATH:-outputs/pretrained/pi05-base-2cam}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-pi05_gem_stack_blocks_three_2cam_folding_recipe_rel30_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}.log}"
USE_PEFT="${USE_PEFT:-true}"
if [[ "${USE_PEFT}" == "true" ]]; then
  OPTIMIZER_LR="${OPTIMIZER_LR:-3.75e-04}"
  SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-05}"
  PEFT_ARGS=(--peft.method_type=LORA --peft.r="${PEFT_R:-64}")
else
  OPTIMIZER_LR="${OPTIMIZER_LR:-3.75e-05}"
  SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-06}"
  PEFT_ARGS=()
fi

mkdir -p "${LOG_DIR}"

if [[ ! -f "${DATASET_ROOT}/meta/stats.json" ]]; then
  echo "Missing relative-action stats dataset at ${DATASET_ROOT}" >&2
  echo "Create it with:" >&2
  echo "  uv run lerobot-edit-dataset --repo_id ${DATASET_REPO} --new_root ${DATASET_ROOT} --operation.type recompute_stats --operation.relative_action true --operation.chunk_size 30 --operation.relative_exclude_joints \"['gripper']\" --operation.num_workers 4" >&2
  exit 1
fi

exec uv run lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model="${COMPILE_MODEL:-false}" \
  --policy.compile_mode=max-autotune \
  --policy.train_expert_only=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.optimizer_lr="${OPTIMIZER_LR}" \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.root="${DATASET_ROOT}" \
  --rename_map='{"observation.images.head":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --steps="${STEPS:-200000}" \
  --batch_size="${BATCH_SIZE:-1}" \
  --eval.batch_size=1 \
  --num_workers="${NUM_WORKERS:-2}" \
  --prefetch_factor=4 \
  --log_freq=50 \
  --eval_freq=5000 \
  --validation_split=0.1 \
  --validation_samples=256 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --wandb.enable=true \
  --wandb.project="${WANDB_PROJECT:-gem-stack-blocks-three-pi05-folding-recipe}" \
  --seed=1000 \
  --job_name="${JOB_NAME}" \
  --output_dir="${OUTPUT_DIR}" \
  "${PEFT_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
