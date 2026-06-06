#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${FEETECH_PORT:?Set FEETECH_PORT, for example /dev/ttyACM0}"

SERVER_ADDRESS="${SERVER_ADDRESS:-100.117.115.124:8080}"
ODRIVE_PORT="${ODRIVE_PORT:-auto}"
ROBOT_ID="${ROBOT_ID:-gem_follower}"
POLICY_PATH="${POLICY_PATH:-outputs/train/pi05_gem_stack_blocks_three_3cam_default_rtc_20260605_132126/checkpoints/200000/pretrained_model}"
TASK="${TASK:-stack the three blocks}"

uv run python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=gem \
  --robot.id="${ROBOT_ID}" \
  --robot.feetech_port="${FEETECH_PORT}" \
  --robot.odrive_port="${ODRIVE_PORT}" \
  --robot.max_relative_target="${MAX_RELATIVE_TARGET:-5}" \
  --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: ${CAM_BASE:-0}, width: ${CAM_WIDTH:-320}, height: ${CAM_HEIGHT:-240}, fps: ${CAM_FPS:-15}}, left_wrist_0_rgb: {type: opencv, index_or_path: ${CAM_WRIST:-1}, width: ${CAM_WIDTH:-320}, height: ${CAM_HEIGHT:-240}, fps: ${CAM_FPS:-15}}}" \
  --task="${TASK}" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${POLICY_PATH}" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk="${ACTIONS_PER_CHUNK:-8}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD:-0.8}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME:-latest_only}" \
  --fps="${CLIENT_FPS:-15}" \
  --debug_visualize_queue_size="${DEBUG_QUEUE:-false}"
