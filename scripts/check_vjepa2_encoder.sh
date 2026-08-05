#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python scripts/check_vjepa2_encoder.py \
  --backend torchhub \
  --hub-repo facebookresearch/vjepa2:main \
  --hub-model vjepa2_1_vit_large_384 \
  --image-size 384 \
  --num-frames 16 \
  --batch-size 2 \
  --print-checkpoint-urls \
  "$@"
