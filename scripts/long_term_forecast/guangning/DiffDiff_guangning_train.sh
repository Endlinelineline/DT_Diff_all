#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-F:/DeepLearning/datasets/electic/风电数据集/guangningshuju}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-100}"
N_SAMPLE="${N_SAMPLE:-20}"

"${PYTHON_BIN}" -s -u run.py train \
  --model DiffDiff \
  --model_id guangning_16turbine \
  --des joint_forecast \
  --root_path "${DATA_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --n_sample "${N_SAMPLE}" \
  "$@"
