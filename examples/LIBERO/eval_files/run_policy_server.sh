#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:?Set CKPT=/path/to/steps_N_pytorch_model.pt}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

your_ckpt=${CKPT}   
gpu_id="${GPU_ID:-0}"
port="${PORT:-6694}"
################# star Policy Server ######################

# export DEBUG=true

M2W_ARGS=""
if [ "${DISABLE_COT_GENERATION:-0}" = "1" ]; then
    M2W_ARGS="${M2W_ARGS} --disable_cot_generation"
fi
if [ -n "${INFERENCE_COT_MAX_NEW_TOKENS:-}" ]; then
    M2W_ARGS="${M2W_ARGS} --inference_cot_max_new_tokens ${INFERENCE_COT_MAX_NEW_TOKENS}"
fi
if [ "${INFERENCE_COT_USE_CACHE:-1}" = "1" ]; then
    M2W_ARGS="${M2W_ARGS} --inference_cot_use_cache"
fi
if [ "${INFERENCE_EMPTY_CACHE:-0}" = "1" ]; then
    M2W_ARGS="${M2W_ARGS} --inference_empty_cache"
fi

CUDA_VISIBLE_DEVICES=$gpu_id ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    ${M2W_ARGS}

# #################################
