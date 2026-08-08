#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-large}"
OUT_DIR="${2:-playground/Pretrained_models/VJEPA2.1}"

case "${MODEL}" in
  base)
    URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
    ;;
  large)
    URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
    ;;
  giant)
    URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt"
    ;;
  gigantic)
    URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt"
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    echo "Usage: bash scripts/download_vjepa2_1_checkpoint.sh [base|large|giant|gigantic] [out_dir]" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${OUT_DIR}" == "torchhub-cache" || "${OUT_DIR}" == "hub-cache" ]]; then
  OUT_DIR="$(python - <<'PY'
try:
    import torch
    print(torch.hub.get_dir() + "/checkpoints")
except Exception:
    import os
    print(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
PY
)"
fi

mkdir -p "${OUT_DIR}"
echo "Downloading V-JEPA2.1 ${MODEL} checkpoint:"
echo "  ${URL}"
echo "to:"
echo "  ${OUT_DIR}"

OUT_FILE="${OUT_DIR}/$(basename "${URL}")"
curl -L -C - "${URL}" -o "${OUT_FILE}"

if [[ "${NO_TORCHHUB_LINK:-0}" != "1" ]]; then
  TORCHHUB_CKPT_DIR="$(python - <<'PY'
try:
    import torch
    print(torch.hub.get_dir() + "/checkpoints")
except Exception:
    import os
    print(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
PY
)"
  mkdir -p "${TORCHHUB_CKPT_DIR}"
  ln -sf "$(cd "$(dirname "${OUT_FILE}")" && pwd)/$(basename "${OUT_FILE}")" \
    "${TORCHHUB_CKPT_DIR}/$(basename "${OUT_FILE}")"
  echo "Linked torchhub cache:"
  echo "  ${TORCHHUB_CKPT_DIR}/$(basename "${OUT_FILE}") -> ${OUT_FILE}"
fi
