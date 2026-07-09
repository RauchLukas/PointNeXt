#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Run up to 4 sweep jobs in parallel (1 GPU each).

CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep0 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep1 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep2 &
wait
