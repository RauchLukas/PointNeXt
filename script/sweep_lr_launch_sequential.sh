#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0001.yaml -n lr0p003__weight_decay0p0001_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0001.yaml -n lr0p003__weight_decay0p0001_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0001.yaml -n lr0p003__weight_decay0p0001_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0005.yaml -n lr0p003__weight_decay0p0005_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0005.yaml -n lr0p003__weight_decay0p0005_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p0005.yaml -n lr0p003__weight_decay0p0005_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p001.yaml -n lr0p003__weight_decay0p001_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p001.yaml -n lr0p003__weight_decay0p001_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p003__weight_decay0p001.yaml -n lr0p003__weight_decay0p001_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0001.yaml -n lr0p005__weight_decay0p0001_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0001.yaml -n lr0p005__weight_decay0p0001_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0001.yaml -n lr0p005__weight_decay0p0001_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0005.yaml -n lr0p005__weight_decay0p0005_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0005.yaml -n lr0p005__weight_decay0p0005_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p0005.yaml -n lr0p005__weight_decay0p0005_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p001.yaml -n lr0p005__weight_decay0p001_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p001.yaml -n lr0p005__weight_decay0p001_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p005__weight_decay0p001.yaml -n lr0p005__weight_decay0p001_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0001.yaml -n lr0p01__weight_decay0p0001_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0001.yaml -n lr0p01__weight_decay0p0001_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0001.yaml -n lr0p01__weight_decay0p0001_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0005.yaml -n lr0p01__weight_decay0p0005_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0005.yaml -n lr0p01__weight_decay0p0005_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p0005.yaml -n lr0p01__weight_decay0p0005_rep2
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p001.yaml -n lr0p01__weight_decay0p001_rep0
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p001.yaml -n lr0p01__weight_decay0p001_rep1
bash "$SCRIPT_DIR/train.sh" -c cfgs/rohbau3d_base/sweep_lr/lr0p01__weight_decay0p001.yaml -n lr0p01__weight_decay0p001_rep2
