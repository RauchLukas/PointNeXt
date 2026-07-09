#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Each run uses 4-GPU DDP; jobs run one after another.

bash "$SCRIPT_DIR/train.sh" -g 4 -c cfgs/rohbau3d_base/sweep_main/batch_size1__voxel_size0p08__voxel_max198000__loop8.yaml -n batch_size1__voxel_size0p08__voxel_max198000__loop8
bash "$SCRIPT_DIR/train.sh" -g 4 -c cfgs/rohbau3d_base/sweep_main/batch_size1__voxel_size0p08__voxel_max198000__loop16.yaml -n batch_size1__voxel_size0p08__voxel_max198000__loop16
