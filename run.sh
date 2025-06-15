#!/bin/bash
set -e  # Exit on error

# Train
python lerobot/scripts/train.py "$@"

# Extract output_dir from args
OUTPUT_DIR=$(echo "$@" | grep -oP '(?<=--output_dir=)[^ ]*')

# Optional safety check
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir must be provided"
    exit 1
fi

# Push to Hugging Face Hub
python lerobot/scripts/push_pretrained.py \
--pretrained_path=$OUTPUT_DIR/checkpoints/last/pretrained_model \
--repo_id=jclinton1/$(basename $OUTPUT_DIR)
