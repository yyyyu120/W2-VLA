#!/bin/bash
#SBATCH --job-name=ebench_baseline
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

# -------------------- NCCL / Networking --------------------
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# -------------------- Required distributed environment --------------------
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 10000))

echo "SLURM_NNODES=$SLURM_NNODES  GPUS_PER_NODE=$GPUS_PER_NODE  TOTAL_GPUS=$TOTAL_GPUS"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# -------------------- Your original config --------------------
Framework_name="${FRAMEWORK_NAME:-QwenOFT}"
freeze_module_list="${FREEZE_MODULE_LIST:-}"
base_vlm="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
config_yaml="${CONFIG_YAML:-./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml}"
run_root_dir="${RUN_ROOT_DIR:-./results/Checkpoints}"
data_mix="${DATA_MIX:-robotwin_all_50}"
run_id="${RUN_ID:-robotwin_all_50_abs_qwen3oft}"

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
fi
conda activate "${CONDA_ENV:-starvla}"

# -------------------- Key: launch accelerate once per node --------------------
srun --jobid "$SLURM_JOBID" bash -c '
  set -e
  echo "Host=$(hostname)  SLURM_PROCID=$SLURM_PROCID"

  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    --machine_rank $SLURM_PROCID \
    --num_machines '"$SLURM_NNODES"' \
    --num_processes '"$TOTAL_GPUS"' \
    starVLA/training/train_starvla.py \
    --config_yaml '"$config_yaml"' \
    --framework.name '"$Framework_name"' \
    --framework.qwenvl.base_vlm '"$base_vlm"' \
    --datasets.vla_data.per_device_batch_size 4 \
    --datasets.vla_data.action_type abs_qpos \
    --datasets.vla_data.action_mode abs \
    --datasets.vla_data.data_mix '"$data_mix"' \
    --trainer.freeze_modules '"$freeze_module_list"' \
    --trainer.max_train_steps 150000 \
    --trainer.save_interval 10000 \
    --trainer.logging_frequency 50 \
    --trainer.eval_interval 1000 \
    --run_root_dir '"$run_root_dir"' \
    --run_id '"$run_id"' \
    --wandb_project '"${WANDB_PROJECT:-w2-vla}"' \
    --wandb_entity '"${WANDB_ENTITY:-}"'
'
