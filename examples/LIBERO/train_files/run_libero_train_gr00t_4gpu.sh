#!/bin/bash

set -euo pipefail

# Source-like StarVLA-GR00T LIBERO baseline entry for a 4-GPU server.
# This uses DeepSpeed ZeRO-2. DeepSpeed checks CUDA_HOME/bin/nvcc at import time,
# so infer CUDA_HOME from the system CUDA toolkit when it is not provided.

export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-online}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export STARVLA_USE_DEEPSPEED=1

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  elif [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda-12.4
  elif command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(readlink -f "$(command -v nvcc)" 2>/dev/null || command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "${nvcc_path}")")"
  else
    echo "Unable to locate nvcc. Set CUDA_HOME to a CUDA toolkit installation." >&2
    exit 1
  fi
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=${Framework_name:-QwenGR00T}
freeze_module_list=${freeze_module_list:-''}
base_vlm=${base_vlm:-./playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/LIBERO/train_files/starvla_cotrain_libero.yaml}
libero_data_root=${libero_data_root:-playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${data_mix:-libero_all}
video_backend=${video_backend:-torchvision_av}
include_state=${include_state:-false}
run_root_dir=${run_root_dir:-./results/Checkpoints}
run_id=${run_id:-starvla_gr00t_libero_zero2_4gpu_effbs128}
num_processes=${num_processes:-4}
per_device_batch_size=${per_device_batch_size:-16}
gradient_accumulation_steps=${gradient_accumulation_steps:-2}
max_train_steps=${max_train_steps:-80000}
num_warmup_steps=${num_warmup_steps:-5000}
repeated_diffusion_steps=${repeated_diffusion_steps:-8}
save_interval=${save_interval:-10000}
logging_frequency=${logging_frequency:-100}
eval_interval=${eval_interval:-100}
wandb_project=${wandb_project:-w2-vla}
wandb_entity=${wandb_entity:-}
# === End of environment variable configuration ===
###########################################################################################

export STARVLA_GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}"

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.video_backend "${video_backend}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.include_state "${include_state}" \
  --datasets.vla_data.sequential_step_sampling False \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${num_warmup_steps}" \
  --trainer.repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  2>&1 | tee "${output_dir}/train.log"
