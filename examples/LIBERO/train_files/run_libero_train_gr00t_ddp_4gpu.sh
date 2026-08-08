#!/bin/bash

set -euo pipefail

# StarVLA-GR00T LIBERO baseline for a 4-GPU DDP setup.
# This avoids DeepSpeed/ZeRO-2 because that machine currently has no CUDA_HOME/nvcc.
# Defaults target 4 GPUs with effective batch size 128:
#   4 GPUs * per_device_batch_size 8 * gradient_accumulation_steps 4.

export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-online}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export STARVLA_USE_DEEPSPEED=0

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
run_id=${run_id:-starvla_gr00t_libero_ddp_4gpu_effbs128}
num_processes=${num_processes:-4}
per_device_batch_size=${per_device_batch_size:-8}
gradient_accumulation_steps=${gradient_accumulation_steps:-4}
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
  --num_processes="${num_processes}" \
  --num_machines=1 \
  --dynamo_backend=no \
  --mixed_precision=bf16 \
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
