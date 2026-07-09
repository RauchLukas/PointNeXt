#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

bash "$SCRIPT_DIR/train.sh" -g 1 -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep0
bash "$SCRIPT_DIR/train.sh" -g 1 -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep1
bash "$SCRIPT_DIR/train.sh" -g 1 -c cfgs/rohbau3d_base/sweep_repeat/baseline.yaml -n baseline_rep2
