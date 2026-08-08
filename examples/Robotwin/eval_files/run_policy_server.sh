#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-1}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

vjepa_fp32_flag=()
if [[ "${ROBOTWIN_VJEPA_FP32:-1}" != "0" ]]; then
    vjepa_fp32_flag+=(--keep_vjepa_fp32)
fi

flow_steps_flag=()
if [[ -n "${ROBOTWIN_FLOW_INFERENCE_STEPS:-}" ]]; then
    if [[ ! "${ROBOTWIN_FLOW_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ROBOTWIN_FLOW_INFERENCE_STEPS must be a positive integer." >&2
        exit 1
    fi
    flow_steps_flag+=(--num_inference_timesteps "${ROBOTWIN_FLOW_INFERENCE_STEPS}")
fi

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"
echo "[INFO] flow inference steps override: ${ROBOTWIN_FLOW_INFERENCE_STEPS:-checkpoint default}"
echo "[INFO] keep V-JEPA FP32: ${ROBOTWIN_VJEPA_FP32:-1}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${use_bf16_flag[@]}" \
    "${vjepa_fp32_flag[@]}" \
    "${flow_steps_flag[@]}"
