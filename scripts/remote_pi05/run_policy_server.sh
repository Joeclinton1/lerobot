#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

uv run python -m lerobot.async_inference.policy_server \
  --host="${LEROBOT_POLICY_SERVER_HOST:-0.0.0.0}" \
  --port="${LEROBOT_POLICY_SERVER_PORT:-8080}" \
  --fps="${LEROBOT_POLICY_FPS:-15}" \
  --inference_latency="${LEROBOT_INFERENCE_LATENCY:-0}" \
  --obs_queue_timeout="${LEROBOT_OBS_QUEUE_TIMEOUT:-0.05}"
