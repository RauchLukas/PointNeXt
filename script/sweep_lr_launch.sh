#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Run up to 4 sweep jobs in parallel (1 GPU each).

CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0001.yaml -n lr0p003__weight_decay0p0001_rep0 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0001.yaml -n lr0p003__weight_decay0p0001_rep1 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0001.yaml -n lr0p003__weight_decay0p0001_rep2 &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0005.yaml -n lr0p003__weight_decay0p0005_rep0 &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0005.yaml -n lr0p003__weight_decay0p0005_rep1 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0005.yaml -n lr0p003__weight_decay0p0005_rep2 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p001.yaml -n lr0p003__weight_decay0p001_rep0 &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p001.yaml -n lr0p003__weight_decay0p001_rep1 &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p001.yaml -n lr0p003__weight_decay0p001_rep2 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0001.yaml -n lr0p005__weight_decay0p0001_rep0 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0001.yaml -n lr0p005__weight_decay0p0001_rep1 &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0001.yaml -n lr0p005__weight_decay0p0001_rep2 &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0005.yaml -n lr0p005__weight_decay0p0005_rep0 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0005.yaml -n lr0p005__weight_decay0p0005_rep1 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0005.yaml -n lr0p005__weight_decay0p0005_rep2 &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p001.yaml -n lr0p005__weight_decay0p001_rep0 &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p001.yaml -n lr0p005__weight_decay0p001_rep1 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p001.yaml -n lr0p005__weight_decay0p001_rep2 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0001.yaml -n lr0p01__weight_decay0p0001_rep0 &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0001.yaml -n lr0p01__weight_decay0p0001_rep1 &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0001.yaml -n lr0p01__weight_decay0p0001_rep2 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0005.yaml -n lr0p01__weight_decay0p0005_rep0 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0005.yaml -n lr0p01__weight_decay0p0005_rep1 &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0005.yaml -n lr0p01__weight_decay0p0005_rep2 &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p001.yaml -n lr0p01__weight_decay0p001_rep0 &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p001.yaml -n lr0p01__weight_decay0p001_rep1 &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p001.yaml -n lr0p01__weight_decay0p001_rep2 &
wait
