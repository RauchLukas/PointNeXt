#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Each run uses 4-GPU DDP; jobs run one after another.

bash "$SCRIPT_DIR/train.sh" -g 4 -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep0
bash "$SCRIPT_DIR/train.sh" -g 4 -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep1
bash "$SCRIPT_DIR/train.sh" -g 4 -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep2
