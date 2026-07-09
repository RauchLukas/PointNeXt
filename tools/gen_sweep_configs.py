"""
Hyperparameter-sweep config generator for PointNeXt (rohbau3D).

PointNeXt configs are YAML files merged recursively by EasyConfig.  Several
values are nested (``dataset.train.loop``, ``dataset.common.voxel_size``, …).
Overriding a single top-level key via CLI does not update those nested uses.
This generator merges the base config chain, applies overrides, and writes
fully self-contained YAML files so every derived value is correct.

Usage:
    # Phase 0 - learning-rate search (reduced loop, fixed batch_size=1):
    python tools/gen_sweep_configs.py --phase lr

    # Phase 1 - main sweep (edit BEST_LR and the grids below first):
    python tools/gen_sweep_configs.py --phase main

    # Repeat study - run the same config N times to verify determinism:
    python tools/gen_sweep_configs.py --phase repeat

Outputs:
    - One config .yaml per setting under cfgs/<DATASET>/<OUT_SUBDIR>/
    - A manifest CSV mapping config name -> parameters
    - Bash launch scripts under script/ (for Linux container / server use)
"""
import argparse
import copy
import csv
import itertools
import os

import yaml

# --------------------------------------------------------------------------- #
# Paths / launch settings (edit to taste)
# --------------------------------------------------------------------------- #
DATASET = "rohbau3d_base"
MODEL_CONFIG = "cfgs/rohbau3d_base/pointnext-xl.yaml"  # XL only
BASE_CONFIGS = [
    "cfgs/default.yaml",
    "cfgs/rohbau3d_base/default.yaml",
    MODEL_CONFIG,
]
# GPUs per training run.  NUM_GPU > 1 enables DDP (mp.spawn) on that many GPUs.
# Sweep jobs are launched sequentially — you cannot run two 4-GPU DDP jobs at once
# on a 4-GPU node.  Set NUM_GPU=1 and NUM_DEVICES=4 for the alternative mode
# (up to 4 independent single-GPU jobs in parallel).
NUM_GPU = 4
NUM_DEVICES = 4

# Nested override paths.  Keys are sweep parameter names; values are dotted
# paths into the merged config dict.
OVERRIDE_PATHS = {
    "lr": ["lr"],
    "weight_decay": ["optimizer", "weight_decay"],
    "batch_size": ["batch_size"],
    "val_batch_size": ["val_batch_size"],
    "loop": ["dataset", "train", "loop"],
    "voxel_size": ["dataset", "common", "voxel_size"],
    "voxel_max": ["dataset", "train", "voxel_max"],
    "val_voxel_max": ["dataset", "val", "voxel_max"],
    "wandb_project": ["wandb", "project"],
    "seed": ["seed"],
    "epochs": ["epochs"],
    "deterministic": ["deterministic"],
    "log_dir": ["log_dir"],
    "metric_fraction": ["metric_fraction"],
    "use_amp": ["use_amp"],
}
NON_NAME_KEYS = {"wandb_project"}

# --------------------------------------------------------------------------- #
# Phase 0: learning-rate search
# --------------------------------------------------------------------------- #
LR_PHASE = dict(
    out_subdir="sweep_lr",
    fixed=dict(
        batch_size=1,
        loop=4,
        voxel_size=0.08,
        voxel_max=96000,
        val_voxel_max=None,  # null = full scene; set e.g. 256000 to cap val memory
        seed=42,
        deterministic=True,
        use_amp=False,
        metric_fraction=True,
        log_dir="rohbau3d_hp_lr",
        epochs=30,
    ),
    grid=dict(
        lr=[0.003, 0.005, 0.01],
        weight_decay=[0.0001, 0.0005, 0.001],
        wandb_project=["PointNeXt-rb3d_hp_lr"],
    ),
    repeats=3,
)

# --------------------------------------------------------------------------- #
# Phase 1: main sweep.  EDIT BEST_LR after Phase 0 finishes.
# --------------------------------------------------------------------------- #
BEST_LR = 0.003
MAIN_PHASE = dict(
    out_subdir="sweep_main",
    fixed=dict(
        lr=BEST_LR,
        weight_decay=0.0001,
        seed=42,
        deterministic=True,
        use_amp=False,
        metric_fraction=True,
        log_dir="rohbau3d_hp_main",
        epochs=50,
        val_voxel_max=256000,
    ),
    grid=dict(
        batch_size=[1],
        voxel_size=[0.08],
        voxel_max=[198000],
        # val_voxel_max=[None, 256000],  # optional: sweep val point cap
        loop=[8, 16],
        wandb_project=["PointNeXt-rb3d_hp_main"],
    ),
    repeats=1,
)

# --------------------------------------------------------------------------- #
# Repeat study: identical config launched N times for determinism checks.
# --------------------------------------------------------------------------- #
REPEAT_PHASE = dict(
    out_subdir="sweep_repeat",
    exp_name="baseline",
    fixed=dict(
        batch_size=1,
        loop=2,
        voxel_size=0.08,
        voxel_max=48000,
        val_voxel_max=None,  # full-scene val for determinism repeat study
        lr=0.01,
        weight_decay=0.0001,
        seed=42,
        deterministic=True,
        use_amp=False,
        metric_fraction=True,
        log_dir="rohbau3d_determinism",
        epochs=5,
        wandb_project="PointNeXt-rb3d_determinism",
    ),
    repeats=3,
)


def deep_merge(base, override):
    """Recursively merge override into base (override wins at leaves)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_merged_config(root, config_paths):
    cfg = {}
    for rel_path in config_paths:
        path = os.path.join(root, rel_path)
        with open(path, "r", encoding="utf-8") as fh:
            partial = yaml.safe_load(fh) or {}
        cfg = deep_merge(cfg, partial)
    return cfg


def set_nested(cfg, path, value):
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key not in OVERRIDE_PATHS:
            raise ValueError(f"'{key}' is not in OVERRIDE_PATHS")
        set_nested(cfg, OVERRIDE_PATHS[key], value)

    # Sweep defaults for reproducible validation.
    cfg.setdefault("seed", 42)
    cfg.setdefault("deterministic", True)
    cfg.setdefault("use_amp", False)
    cfg.setdefault("metric_fraction", True)
    cfg.setdefault("wandb", {})
    cfg["wandb"]["use_wandb"] = True
    cfg.setdefault("dataset", {}).setdefault("test", {})["voxel_max"] = None
    if "val_voxel_max" not in overrides:
        cfg.setdefault("dataset", {}).setdefault("val", {})["voxel_max"] = None
    cfg["val_batch_size"] = 1  # variable point counts per scene — must not batch >1
    cfg["world_size"] = NUM_GPU
    cfg["ngpus_per_node"] = NUM_GPU
    cfg["multiprocessing_distributed"] = NUM_GPU > 1
    return cfg


def _tag(value):
    if value is None:
        return "null"
    return str(value).replace(".", "p").replace("-", "m")


def dump_config(cfg, path):
    header = (
        "# Auto-generated by tools/gen_sweep_configs.py\n"
        "# Self-contained sweep config (merged from cfgs/default.yaml chain).\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)


def _launch_cmd(config_path, wandb_name=""):
    gpu_flag = f" -g {NUM_GPU}" if NUM_GPU > 1 else ""
    if wandb_name:
        return f'bash "$SCRIPT_DIR/train.sh"{gpu_flag} -c {config_path} -n {wandb_name}'
    return f'bash "$SCRIPT_DIR/train.sh"{gpu_flag} -c {config_path}'


def _launch_header():
    return [
        "#!/bin/sh",
        "set -e",
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"',
        'cd "$ROOT_DIR"',
        "",
    ]


def _wandb_name(row):
    if row.get("repeats", 1) > 1:
        return f"{row['exp_name']}_rep{row['replicate_id']}"
    return row["exp_name"]


def write_launch_scripts(root, out_subdir, manifest_rows):
    """Write sequential and (when possible) parallel launch scripts."""
    seq_lines = _launch_header()
    if NUM_GPU > 1:
        seq_lines.append(f"# Each run uses {NUM_GPU}-GPU DDP; jobs run one after another.")
        seq_lines.append("")
    for row in manifest_rows:
        seq_lines.append(_launch_cmd(row["config_path"], _wandb_name(row)))

    seq_path = os.path.join(root, "script", f"{out_subdir}_launch_sequential.sh")
    with open(seq_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(seq_lines) + "\n")
    os.chmod(seq_path, 0o755)

    use_parallel = NUM_GPU == 1 and NUM_DEVICES > 1 and len(manifest_rows) > 1
    if use_parallel:
        par_lines = _launch_header() + [
            f"# Run up to {NUM_DEVICES} sweep jobs in parallel (1 GPU each).",
            "",
        ]
        for i, row in enumerate(manifest_rows):
            device = i % NUM_DEVICES
            par_lines.append(
                f'CUDA_VISIBLE_DEVICES={device} bash "$SCRIPT_DIR/train.sh" '
                f'-c {row["config_path"]} -n {_wandb_name(row)} &'
            )
            if device == NUM_DEVICES - 1:
                par_lines.append("wait")
        par_lines.append("wait")
        primary_path = os.path.join(root, "script", f"{out_subdir}_launch.sh")
        with open(primary_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(par_lines) + "\n")
        os.chmod(primary_path, 0o755)
        print(f"[{out_subdir}] launch (parallel, {NUM_DEVICES} GPUs) -> {primary_path}")
        print(f"[{out_subdir}] launch (sequential)            -> {seq_path}")
        return primary_path, seq_path

    primary_path = os.path.join(root, "script", f"{out_subdir}_launch.sh")
    with open(primary_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(seq_lines) + "\n")
    os.chmod(primary_path, 0o755)
    if NUM_GPU > 1:
        print(f"[{out_subdir}] launch (sequential, {NUM_GPU}-GPU DDP) -> {primary_path}")
    else:
        print(f"[{out_subdir}] launch script -> {primary_path}")
    return primary_path, None


def build_grid_phase(phase_cfg, base_cfg, root):
    out_subdir = phase_cfg["out_subdir"]
    fixed = phase_cfg["fixed"]
    grid = phase_cfg.get("grid", {})
    repeats = int(phase_cfg.get("repeats", 1))

    out_dir = os.path.join(root, "cfgs", DATASET, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    grid_keys = list(grid.keys())
    combos = [()] if not grid_keys else list(itertools.product(*[grid[k] for k in grid_keys]))

    manifest_rows = []
    for combo in combos:
        overrides = dict(fixed)
        overrides.update({k: v for k, v in zip(grid_keys, combo)})

        name_tokens = [
            f"{k}{_tag(v)}" for k, v in zip(grid_keys, combo) if k not in NON_NAME_KEYS
        ]
        exp_name = "__".join(name_tokens) if name_tokens else "run"
        config_rel = f"cfgs/{DATASET}/{out_subdir}/{exp_name}.yaml"
        config_path = os.path.join(out_dir, f"{exp_name}.yaml")

        cfg = apply_overrides(base_cfg, overrides)
        dump_config(cfg, config_path)

        for rep in range(repeats):
            row = dict(
                config_path=config_rel,
                exp_name=exp_name,
                replicate_id=rep,
                repeats=repeats,
            )
            row.update(overrides)
            manifest_rows.append(row)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = sorted({k for row in manifest_rows for k in row.keys()})
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_launch_scripts(root, out_subdir, manifest_rows)

    print(f"[{out_subdir}] wrote {len(combos)} config(s), {len(manifest_rows)} launch row(s) -> {out_dir}")
    print(f"[{out_subdir}] manifest -> {manifest_path}")


def build_repeat_phase(phase_cfg, base_cfg, root):
    out_subdir = phase_cfg["out_subdir"]
    exp_name = phase_cfg["exp_name"]
    overrides = dict(phase_cfg["fixed"])
    repeats = int(phase_cfg["repeats"])

    out_dir = os.path.join(root, "cfgs", DATASET, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    config_rel = f"cfgs/{DATASET}/{out_subdir}/{exp_name}.yaml"
    config_path = os.path.join(out_dir, f"{exp_name}.yaml")
    cfg = apply_overrides(base_cfg, overrides)
    dump_config(cfg, config_path)

    readme_path = os.path.join(out_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(
            "Determinism repeat study\n"
            "========================\n"
            f"Config: {config_rel}\n"
            f"Repeats: {repeats}\n"
            f"Seed: {overrides.get('seed', 42)}\n"
            f"Deterministic: {overrides.get('deterministic', True)}\n\n"
            "Launch (inside container / on server):\n"
            f"  bash script/{out_subdir}_launch.sh  # sequential, {NUM_GPU}-GPU DDP per run\n\n"
            "If training is deterministic, every replicate should report the same\n"
            "val mIoU per epoch (within tiny float noise). Compare wandb runs or\n"
            f"the CSV files written under log/{DATASET}/.\n"
        )

    manifest_rows = []
    for rep in range(repeats):
        row = dict(
            replicate_id=rep,
            config_path=config_rel,
            exp_name=exp_name,
            repeats=repeats,
        )
        row.update(overrides)
        manifest_rows.append(row)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = sorted({k for row in manifest_rows for k in row.keys()})
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_launch_scripts(root, out_subdir, manifest_rows)

    print(f"[{out_subdir}] wrote 1 config, {repeats} replicate launch(es)")
    print(f"[{out_subdir}] manifest -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate PointNeXt rohbau3D sweep configs")
    parser.add_argument(
        "--phase",
        choices=["lr", "main", "repeat"],
        required=True,
        help="which sweep phase to generate",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_cfg = load_merged_config(root, BASE_CONFIGS)

    if args.phase == "lr":
        build_grid_phase(LR_PHASE, base_cfg, root)
    elif args.phase == "main":
        build_grid_phase(MAIN_PHASE, base_cfg, root)
    else:
        build_repeat_phase(REPEAT_PHASE, base_cfg, root)


if __name__ == "__main__":
    main()
