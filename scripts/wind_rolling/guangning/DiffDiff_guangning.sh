#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-F:/DeepLearning/datasets/electic/风电数据集/guangningshuju}"
N_SAMPLE="${N_SAMPLE:-20}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a .ckpt file or checkpoint directory}"

"${PYTHON_BIN}" -s -u run.py rolling15 \
  --model DiffDiff \
  --root_path "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --release_interval 15 \
  --n_sample "${N_SAMPLE}" \
  "$@"

