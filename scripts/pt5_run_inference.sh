#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 INPUT_DIR MODEL_CHECKPOINT [GPU]"
    echo "  INPUT_DIR        Folder with tiles (e.g. AP_*.tif)"
    echo "  MODEL_CHECKPOINT Path to FTW .ckpt file"
    echo "  GPU              Optional GPU index (default: 0)"
    exit 1
fi

INPUT_DIR="$1"
MODEL_CHECKPOINT="$2"
GPU="${3:-0}"

# Adjust PYTHONPATH if needed depending on where 'src' lives in your environment.
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" \
python -m trazo.pt5_inference.batch_inference \
    --input-dir "$INPUT_DIR" \
    --model-checkpoint "$MODEL_CHECKPOINT" \
    --gpu "$GPU"
