#!/usr/bin/env bash
set -euo pipefail

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens5f0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_SILENT=${WANDB_SILENT:-true}

use_deepspeed=${use_deepspeed:-true}
if [[ "${use_deepspeed}" == "true" || "${use_deepspeed}" == "1" ]]; then
  export STARVLA_USE_DEEPSPEED=1
  accelerate_config=${accelerate_config:-starVLA/config/deepseeds/deepspeed_zero2.yaml}
else
  export STARVLA_USE_DEEPSPEED=0
  accelerate_config=${accelerate_config:-}
fi

# RoboTwin clean-only, 16-step GR00T action chunk, plus implicit CoT query
# supervision. Uses the Robotwin M2W dataloader so later wrist-history and
# future-wrist branches share the same sample contract. DiT receives Qwen
# image/task/query hidden states, not the teacher-forced CoT answer text.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml}
run_root_dir=${run_root_dir:-./results/Checkpoints}
data_mix=${data_mix:-robotwin_clean_h16}
subtask_label_dir=${subtask_label_dir:-${STARVLA_DIR}/playground/Datasets/W2-VLA-CoT/robotwin}
cot_prompt_version=${cot_prompt_version:-robotwin_tri_v1}
cot_prompt=${COT_PROMPT:-$'Your task is: {instruction} Observation: using the current high camera, left wrist camera, and right wrist camera images, identify the robot'\''s current manipulation step.\n\nOutput exactly:\nSubtask: ...\nReasoning: ...\nWrist: left=...; right=...\n\nDo not add extra text.'}
cot_prompt_cli=$(
  COT_PROMPT_TEXT="${cot_prompt}" python - <<'PY'
import json
import os

print(json.dumps(os.environ["COT_PROMPT_TEXT"]))
PY
)
run_id=${run_id:-robotwin_clean_h16_implicit_cot_q16}

num_processes=${NUM_PROCESSES:-4}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-4}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}
max_train_steps=${MAX_TRAIN_STEPS:-100000}
num_warmup_steps=${NUM_WARMUP_STEPS:-5000}
save_interval=${SAVE_INTERVAL:-10000}
logging_frequency=${LOGGING_FREQUENCY:-100}
eval_interval=${EVAL_INTERVAL:-1000}
num_workers=${NUM_WORKERS:-3}
latent_cot_num_queries=${LATENT_COT_NUM_QUERIES:-16}
lambda_cot=${LAMBDA_COT:-0.1}
wrist_history_frames=${WRIST_HISTORY_FRAMES:-1}
load_wrist_future_views=${LOAD_WRIST_FUTURE_VIEWS:-false}
future_wrist_k=${FUTURE_WRIST_K:-8}
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

echo "[Robotwin implicit CoT] CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "[Robotwin implicit CoT] python=$(command -v python)"
echo "[Robotwin implicit CoT] accelerate=${accelerate_bin}"
echo "[Robotwin implicit CoT] label_dir=${subtask_label_dir}"

accelerate_args=(launch --num_processes "${num_processes}")
if [[ -n "${accelerate_config}" ]]; then
  accelerate_args+=(--config_file "${accelerate_config}")
fi

"${accelerate_bin}" "${accelerate_args[@]}" \
  starVLA/training/train_subtask_m2w.py \
  --config_yaml "${config_yaml}" \
  --framework.name QwenSubtaskM2W \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.main_to_wrist.cot_reasoning_mode implicit_query \
  --framework.main_to_wrist.latent_cot_num_queries "${latent_cot_num_queries}" \
  --framework.main_to_wrist.use_cot_loss true \
  --framework.main_to_wrist.use_wrist_future_loss false \
  --framework.main_to_wrist.lambda_cot "${lambda_cot}" \
  --framework.main_to_wrist.lambda_wrist 0.0 \
  --framework.main_to_wrist.generate_cot_at_inference false \
  --framework.main_to_wrist.action_condition_mode qwen_only \
  --framework.main_to_wrist.qwen_input_views all \
  --framework.main_to_wrist.qwen_action_context_mode exclude_cot_prompt \
  --framework.main_to_wrist.skip_vjepa_init true \
  --framework.action_model.action_horizon 16 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.dataset_py subtask_m2w_robotwin_datasets \
  --datasets.vla_data.data_root_dir playground/Datasets/RoboTwin \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.action_type abs_qpos \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.num_workers "${num_workers}" \
  --datasets.vla_data.subtask_label_dir "${subtask_label_dir}" \
  --datasets.vla_data.cot_prompt_version "${cot_prompt_version}" \
  --datasets.vla_data.CoT_prompt "${cot_prompt_cli}" \
  --datasets.vla_data.wrist_history_frames "${wrist_history_frames}" \
  --datasets.vla_data.load_wrist_future_views "${load_wrist_future_views}" \
  --datasets.vla_data.future_wrist_k "${future_wrist_k}" \
  --trainer.train_qwen_vl true \
  --trainer.find_unused_parameters false \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${num_warmup_steps}" \
  --trainer.repeated_diffusion_steps 8 \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --trainer.learning_rate.vlm 1.0e-05 \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --trainer.save_optimizer_state true \
  --trainer.resume_optimizer_state false \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  "$@"
