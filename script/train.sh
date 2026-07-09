#!/usr/bin/env bash
# Container/server training entry point.
# Location-independent: invoke from anywhere; always cd's to the repo root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CFG=""
RUN_NAME=""
NUM_GPU=""

usage() {
    echo "Usage: $0 -c CONFIG [-n WANDB_NAME] [-g NUM_GPU] [-- extra args]"
    echo "   or: $0 CONFIG [extra args...]"
    exit 1
}

while getopts "c:n:g:h" opt; do
    case $opt in
        c) CFG="$OPTARG" ;;
        n) RUN_NAME="$OPTARG" ;;
        g) NUM_GPU="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [[ -z "$CFG" && $# -gt 0 ]]; then
    CFG="$1"
    shift
fi

if [[ -z "$CFG" ]]; then
    echo "Error: config path required (-c CONFIG)" >&2
    usage
fi

# Default: 1 GPU unless -g is passed.  Sweep launch scripts pass -g explicitly.
if [[ -z "$NUM_GPU" ]]; then
    NUM_GPU=1
fi

if [[ "$NUM_GPU" -eq 1 ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES=0
    fi
    EXTRA=(world_size=1 ngpus_per_node=1 multiprocessing_distributed=False)
else
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPU - 1)))"
    fi
    EXTRA=(
        world_size="$NUM_GPU"
        ngpus_per_node="$NUM_GPU"
        multiprocessing_distributed=True
    )
fi

EXTRA+=("$@")
if [[ -n "$RUN_NAME" ]]; then
    EXTRA+=("wandb.name=$RUN_NAME")
fi

exec python examples/segmentation/main.py --cfg "$CFG" "${EXTRA[@]}"
