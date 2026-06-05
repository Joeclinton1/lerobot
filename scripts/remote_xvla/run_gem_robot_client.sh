#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${FEETECH_PORT:?Set FEETECH_PORT, for example /dev/ttyACM0}"

SERVER_ADDRESS="${SERVER_ADDRESS:-100.117.115.124:8080}"
ODRIVE_PORT="${ODRIVE_PORT:-auto}"
ROBOT_ID="${ROBOT_ID:-gem_follower}"
POLICY_PATH="${POLICY_PATH:-outputs/train/xvla_gem_stack_blocks_three_200k_20260602_061641/checkpoints/200000/pretrained_model}"
TASK="${TASK:-stack the three blocks}"

uv run python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=gem \
  --robot.id="${ROBOT_ID}" \
  --robot.feetech_port="${FEETECH_PORT}" \
  --robot.odrive_port="${ODRIVE_PORT}" \
  --robot.max_relative_target=5 \
  --robot.cameras="{ image: {type: opencv, index_or_path: ${CAM_IMAGE:-0}, width: 320, height: 240, fps: 15}, image2: {type: opencv, index_or_path: ${CAM_IMAGE2:-1}, width: 320, height: 240, fps: 15}}" \
  --task="${TASK}" \
  --policy_type=xvla \
  --pretrained_name_or_path="${POLICY_PATH}" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk="${ACTIONS_PER_CHUNK:-8}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD:-0.8}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME:-latest_only}" \
  --fps="${CLIENT_FPS:-15}" \
  --debug_visualize_queue_size="${DEBUG_QUEUE:-true}"
