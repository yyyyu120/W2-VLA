#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-starVLA/config/training/subtask_m2w_libero.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

accelerate launch starVLA/training/train_subtask_m2w.py \
  --config_yaml "${CONFIG}" \
  "$@"
