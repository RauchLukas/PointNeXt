#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Run up to 4 sweep jobs in parallel (1 GPU each).

CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_main/batch_size1__voxel_size0p08__voxel_max198000__loop8.yaml -n batch_size1__voxel_size0p08__voxel_max198000__loop8 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_main/batch_size1__voxel_size0p08__voxel_max198000__loop16.yaml -n batch_size1__voxel_size0p08__voxel_max198000__loop16 &
wait
