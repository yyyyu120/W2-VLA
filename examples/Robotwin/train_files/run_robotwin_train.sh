#!/usr/bin/env bash
set -euo pipefail

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name="${FRAMEWORK_NAME:-QwenOFT}"
freeze_module_list="${FREEZE_MODULE_LIST:-}"
base_vlm="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
config_yaml="${CONFIG_YAML:-./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml}"
run_root_dir="${RUN_ROOT_DIR:-./results/Checkpoints}"
data_mix="${DATA_MIX:-robotwin_all_50}"
run_id="${RUN_ID:-robotwin_all_50_qwen3oft}"
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
# mv this script to the output dir
cp "$0" "${output_dir}/"


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-4}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-150000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-10000}" \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${WANDB_PROJECT:-w2-vla}" \
  --wandb_entity "${WANDB_ENTITY:-}"



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
