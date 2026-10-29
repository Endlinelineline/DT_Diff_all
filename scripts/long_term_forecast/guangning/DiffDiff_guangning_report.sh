#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-F:/DeepLearning/datasets/electic/风电数据集/guangningshuju}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a .ckpt file or checkpoint directory}"
PREDICTIONS="${PREDICTIONS:?Set PREDICTIONS to the saved kW predictions .npy}"
TRUTH="${TRUTH:?Set TRUTH to the saved kW truth .npy}"

"${PYTHON_BIN}" -s -u run.py report \
  --model DiffDiff \
  --root_path "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --predictions "${PREDICTIONS}" \
  --truth "${TRUTH}" \
  --source_layout NHT \
  "$@"
