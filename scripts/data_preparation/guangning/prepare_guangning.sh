#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-F:/DeepLearning/datasets/electic/风电数据集/guangningshuju}"

"${PYTHON_BIN}" -s -u run.py prepare --root_path "${DATA_ROOT}" "$@"

