#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot-gem}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  # shellcheck disable=SC1091
  source /home/joe/miniforge3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

DATASET_REPO="${DATASET_REPO:-jclinton1/gem_stack_blocks_three_20260502_194746}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/gem_stack_blocks_three_rel30_stats}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-allenai/MolmoAct2}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-molmoact2_gem_stack_blocks_three_1gpu_b3_fullft_paged_bnb_100k_val2k_ckpt20k_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}.log}"

BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-20}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${DATASET_ROOT}/meta/stats.json" ]]; then
  echo "Missing relative-action stats dataset at ${DATASET_ROOT}" >&2
  echo "This MolmoAct2 recipe uses the same rel30 stats dataset as the Pi0.5 run." >&2
  echo "Create or restore it before training. The Pi0.5 script used:" >&2
  echo "  lerobot-edit-dataset --repo_id ${DATASET_REPO} --new_root ${DATASET_ROOT} --operation.type recompute_stats --operation.relative_action true --operation.chunk_size 30 --operation.relative_exclude_joints \"['gripper']\" --operation.num_workers 4" >&2
  exit 1
fi

exec lerobot-train \
  --policy.type=molmoact2 \
  --policy.checkpoint_path="${POLICY_CHECKPOINT}" \
  --policy.device=cuda \
  --policy.model_dtype=bfloat16 \
  --policy.action_mode=continuous \
  --policy.inference_action_mode=continuous \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.setup_type="single GEM robotic arm with base and wrist RGB cameras" \
  --policy.control_mode="relative joint pose with absolute gripper command" \
  --policy.image_keys='["observation.images.base_0_rgb","observation.images.left_wrist_0_rgb"]' \
  --policy.num_flow_timesteps=8 \
  --policy.gradient_checkpointing=true \
  --policy.enable_lora_vlm=false \
  --policy.enable_lora_action_expert=false \
  --policy.freeze_embedding=true \
  --policy.normalize_gripper=true \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.use_bnb_optimizer=true \
  --policy.optimizer_paged=true \
  --policy.optimizer_lr="${OPTIMIZER_LR:-1e-5}" \
  --policy.optimizer_vit_lr="${OPTIMIZER_VIT_LR:-5e-6}" \
  --policy.optimizer_connector_lr="${OPTIMIZER_CONNECTOR_LR:-5e-6}" \
  --policy.optimizer_action_expert_lr="${OPTIMIZER_ACTION_EXPERT_LR:-5e-5}" \
  --policy.optimizer_betas='[0.9, 0.95]' \
  --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay="${OPTIMIZER_WEIGHT_DECAY:-0.0}" \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS:-1000}" \
  --policy.scheduler_decay_steps="${STEPS}" \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR:-1e-6}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable="${IMAGE_TRANSFORMS_ENABLE:-true}" \
  --rename_map='{"observation.images.head":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --num_workers="${NUM_WORKERS}" \
  --prefetch_factor=4 \
  --log_freq="${LOG_FREQ}" \
  --eval_freq="${EVAL_FREQ}" \
  --validation_split=0.1 \
  --validation_samples=256 \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable="${WANDB_ENABLE:-true}" \
  --wandb.project="${WANDB_PROJECT:-gem-stack-blocks}" \
  --wandb.group="${WANDB_GROUP:-gem-stack-blocks-three-molmoact2-fullft-rtc}" \
  --seed=1000 \
  --job_name="${JOB_NAME}" \
  --output_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
