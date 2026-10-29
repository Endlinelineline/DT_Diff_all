#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
: "${CUDA_VISIBLE_DEVICES:?Set one free physical GPU before launching}"
: "${SETTING:?Set a unique experiment setting}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/artifacts/python_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset/Guangning}"
LOG_FILE="$PROJECT_ROOT/logs/${SETTING}.log"
EXIT_FILE="$PROJECT_ROOT/logs/${SETTING}.exit_code"
mkdir -p "$PROJECT_ROOT/logs"
if [[ -e "$LOG_FILE" || -e "$EXIT_FILE" || -e "$PROJECT_ROOT/checkpoints/$SETTING" ]]; then
    echo "Refusing to overwrite an existing experiment: $SETTING" >&2
    exit 2
fi

"$PYTHON_BIN" -s -u run.py train \
    --model DiffDiff \
    --model_id guangning_16turbine \
    --des full_first_second_order \
    --setting "$SETTING" \
    --root_path "$DATA_ROOT" \
    --gpu 0 \
    --batch_size 16 \
    --epochs 100 \
    --n_sample 20 \
    "$@" >"$LOG_FILE" 2>&1
status=$?
printf '%s\n' "$status" >"$EXIT_FILE"
exit "$status"
