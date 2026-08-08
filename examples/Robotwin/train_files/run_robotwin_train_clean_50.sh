#!/usr/bin/env bash
set -euo pipefail

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens5f0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_SILENT=${WANDB_SILENT:-true}

# Clean-only RoboTwin, 50-step action chunk.
# The legacy "robotwin" data_mix remains 16-step; "robotwin_clean_50" uses the
# same clean task directories with robotwin50 action_indices.
Framework_name=${Framework_name:-QwenOFT}
freeze_module_list=${freeze_module_list:-''}
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/Robotwin/train_files/starvla_cotrain_robotwin_clean_50.yaml}
run_root_dir=${run_root_dir:-./results/Checkpoints}
data_mix=${data_mix:-robotwin_clean_50}
run_id=${run_id:-robotwin_clean_50chunk_qwen3OFT}

num_processes=${NUM_PROCESSES:-4}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-4}
max_train_steps=${MAX_TRAIN_STEPS:-150000}
save_interval=${SAVE_INTERVAL:-10000}
logging_frequency=${LOGGING_FREQUENCY:-100}
eval_interval=${EVAL_INTERVAL:-1000}
wandb_project=${WANDB_PROJECT:-w2-vla}
wandb_entity=${WANDB_ENTITY:-}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/accelerate" ]]; then
  accelerate_bin="${CONDA_PREFIX}/bin/accelerate"
else
  accelerate_bin="$(command -v accelerate)"
fi

echo "[Robotwin train] CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "[Robotwin train] python=$(command -v python)"
echo "[Robotwin train] accelerate=${accelerate_bin}"

"${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_horizon 50 \
  --framework.action_model.future_action_window_size 49 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir playground/Datasets/RoboTwin \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.action_type abs_qpos \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  "$@"
