#!/bin/sh
set -e
# Location-independent: invoke from anywhere (e.g. the directory above
# Pointcept). train.sh itself cd's to the Pointcept root, so data_root
# (../data) resolves correctly regardless of where you run this from.
SCRIPT_DIR=$(dirname "$0")

sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_main/batch_size12__grid_size0p02__voxel_max102400__loop16 -n sweep_main/batch_size12__grid_size0p02__voxel_max102400__loop16 -g 4
