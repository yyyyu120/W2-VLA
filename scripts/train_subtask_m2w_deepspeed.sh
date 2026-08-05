#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Easy-edit settings
# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/train_subtask_m2w_deepspeed.sh CONFIG.yaml
#
# Common configs:
#   starVLA/config/training/subtask_m2w_libero_w2.yaml
#   starVLA/config/training/subtask_m2w_robotwin_w2.yaml
#
# Any variable below can also be overridden from the shell, for example:
#   NUM_PROCESSES=1 PER_DEVICE_BATCH_SIZE=1 \
#   bash scripts/train_subtask_m2w_deepspeed.sh starVLA/config/training/subtask_m2w_robotwin_w2.yaml

CONFIG=${1:-starVLA/config/training/subtask_m2w_robotwin_w2.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

# Launcher.
USE_DEEPSPEED=${USE_DEEPSPEED:-${use_deepspeed:-true}}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-${num_processes:-8}}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-${gradient_accumulation_steps:-1}}

# Resume. Default is off.
RESUME=${RESUME:-${resume:-${is_resume:-false}}}
RESUME_OPTIMIZER_STATE=${RESUME_OPTIMIZER_STATE:-${resume_optimizer_state:-${RESUME}}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${checkpoint_dir:-}}
RESUME_STEP=${RESUME_STEP:-${resume_step:-}}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${pretrained_checkpoint:-}}
RELOAD_MODULES=${RELOAD_MODULES:-${reload_modules:-}}

# Dataset / model overrides. Leave empty to use the YAML value.
BASE_VLM=${BASE_VLM:-${base_vlm:-}}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-${data_root_dir:-${robotwin_data_root:-${libero_data_root:-}}}}
DATA_MIX=${DATA_MIX:-${data_mix:-}}
DATASET_PY=${DATASET_PY:-${dataset_py:-}}
SUBTASK_LABEL_DIR=${SUBTASK_LABEL_DIR:-${subtask_label_dir:-}}
COT_PROMPT_VERSION=${COT_PROMPT_VERSION:-${cot_prompt_version:-}}
COT_PROMPT=${COT_PROMPT:-}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-${per_device_batch_size:-}}
NUM_WORKERS=${NUM_WORKERS:-${num_workers:-}}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-${prefetch_factor:-}}
SKIP_MISSING_DATASETS=${SKIP_MISSING_DATASETS:-${skip_missing_datasets:-}}
INCLUDE_STATE=${INCLUDE_STATE:-${include_state:-}}
VLM_INCLUDE_WRIST_VIEW=${VLM_INCLUDE_WRIST_VIEW:-${vlm_include_wrist_view:-}}
WRIST_HISTORY_FRAMES=${WRIST_HISTORY_FRAMES:-${wrist_history_frames:-}}
FUTURE_WRIST_FRAMES=${FUTURE_WRIST_FRAMES:-${future_wrist_frames:-}}
LOAD_WRIST_FUTURE_VIEWS=${LOAD_WRIST_FUTURE_VIEWS:-${load_wrist_future_views:-}}
FUTURE_WRIST_K=${FUTURE_WRIST_K:-${future_wrist_k:-}}
FUTURE_WRIST_SMALL_GAP=${FUTURE_WRIST_SMALL_GAP:-${future_wrist_small_gap:-}}
ACTION_TYPE=${ACTION_TYPE:-${action_type:-}}
ACTION_MODE=${ACTION_MODE:-${action_mode:-}}
VIDEO_BACKEND=${VIDEO_BACKEND:-${video_backend:-}}

# Reasoning / JEPA overrides.
COT_REASONING_MODE=${COT_REASONING_MODE:-${cot_reasoning_mode:-}}
LATENT_COT_NUM_QUERIES=${LATENT_COT_NUM_QUERIES:-${latent_cot_num_queries:-}}
QWEN_ACTION_CONTEXT_MODE=${QWEN_ACTION_CONTEXT_MODE:-${qwen_action_context_mode:-}}
LAMBDA_COT=${LAMBDA_COT:-${lambda_cot:-}}
GENERATE_COT_AT_INFERENCE=${GENERATE_COT_AT_INFERENCE:-${generate_cot_at_inference:-}}
JEPA_PREDICTION_VIEW=${JEPA_PREDICTION_VIEW:-${jepa_prediction_view:-}}
LAMBDA_JEPA=${LAMBDA_JEPA:-${lambda_jepa:-}}
USE_JEPA_LOSS=${USE_JEPA_LOSS:-${use_jepa_loss:-}}
NUM_WRIST_VIEWS=${NUM_WRIST_VIEWS:-${num_wrist_views:-}}
JEPA_PREDICTOR_DEPTH=${JEPA_PREDICTOR_DEPTH:-${jepa_predictor_depth:-}}
JEPA_PREDICTOR_NUM_HEADS=${JEPA_PREDICTOR_NUM_HEADS:-${jepa_predictor_num_heads:-}}
JEPA_PREDICTOR_ACTIVATION_CHECKPOINTING=${JEPA_PREDICTOR_ACTIVATION_CHECKPOINTING:-${jepa_predictor_activation_checkpointing:-}}
WRIST_ACTION_CONTEXT_TOKENS=${WRIST_ACTION_CONTEXT_TOKENS:-${wrist_action_context_tokens:-}}
ACTION_HEAD_GRADIENT_CHECKPOINTING=${ACTION_HEAD_GRADIENT_CHECKPOINTING:-${action_head_gradient_checkpointing:-}}

# Action / trainer overrides.
ACTION_HORIZON=${ACTION_HORIZON:-${action_horizon:-}}
FUTURE_ACTION_WINDOW_SIZE=${FUTURE_ACTION_WINDOW_SIZE:-${future_action_window_size:-}}
PAST_ACTION_WINDOW_SIZE=${PAST_ACTION_WINDOW_SIZE:-${past_action_window_size:-}}
ACTION_DIM=${ACTION_DIM:-${action_dim:-}}
STATE_DIM=${STATE_DIM:-${state_dim:-}}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-${max_train_steps:-}}
NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS:-${num_warmup_steps:-}}
REPEATED_DIFFUSION_STEPS=${REPEATED_DIFFUSION_STEPS:-${repeated_diffusion_steps:-}}
DIT_DROPOUT=${DIT_DROPOUT:-${dit_dropout:-}}
TRAIN_QWEN_VL=${TRAIN_QWEN_VL:-${train_qwen_vl:-}}
FIND_UNUSED_PARAMETERS=${FIND_UNUSED_PARAMETERS:-${find_unused_parameters:-}}
SAVE_INTERVAL=${SAVE_INTERVAL:-${save_interval:-}}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-${logging_frequency:-}}
EVAL_INTERVAL=${EVAL_INTERVAL:-${eval_interval:-}}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-${run_root_dir:-}}
RUN_ID=${RUN_ID:-${run_id:-}}

# W&B credentials must come from the environment or `wandb login`.
WANDB_API_KEY=${WANDB_API_KEY:-${wandb_key:-}}
WANDB_MODE=${WANDB_MODE:-${wandb_mode:-offline}}
WANDB_PROJECT=${WANDB_PROJECT:-${wandb_project:-w2-vla}}
WANDB_ENTITY=${WANDB_ENTITY:-${wandb_entity:-}}

# ---------------------------------------------------------------------------
# Launch assembly. Usually no need to edit below this line.
# ---------------------------------------------------------------------------

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export WANDB_MODE
if [[ -n "${WANDB_API_KEY}" ]]; then
  export WANDB_API_KEY
fi

if [[ "${USE_DEEPSPEED}" != "true" && "${USE_DEEPSPEED}" != "1" ]]; then
  ACCELERATE_CONFIG=""
fi

if [[ "${skip_vjepa_init:-false}" == "true" || "${skip_vjepa_init:-0}" == "1" ]]; then
  echo "[W2] V-JEPA is required for W2 future-latent training." >&2
  echo "[W2] Do not set skip_vjepa_init=true." >&2
  exit 1
fi

if [[ -z "${PRETRAINED_CHECKPOINT}" && -n "${CHECKPOINT_DIR}" && -n "${RESUME_STEP}" ]]; then
  PRETRAINED_CHECKPOINT="${CHECKPOINT_DIR}/steps_${RESUME_STEP}_pytorch_model.pt"
fi

json_cot_prompt=""
if [[ -n "${COT_PROMPT}" ]]; then
  json_cot_prompt=$(
    COT_PROMPT_TEXT="${COT_PROMPT}" python - <<'PY'
import json
import os

print(json.dumps(os.environ["COT_PROMPT_TEXT"]))
PY
  )
fi

override_args=()
add_override() {
  local key="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    override_args+=("${key}" "${value}")
  fi
}

add_override "--framework.qwenvl.base_vlm" "${BASE_VLM}"
add_override "--datasets.vla_data.data_root_dir" "${DATA_ROOT_DIR}"
add_override "--datasets.vla_data.data_mix" "${DATA_MIX}"
add_override "--datasets.vla_data.dataset_py" "${DATASET_PY}"
add_override "--datasets.vla_data.subtask_label_dir" "${SUBTASK_LABEL_DIR}"
add_override "--datasets.vla_data.cot_prompt_version" "${COT_PROMPT_VERSION}"
add_override "--datasets.vla_data.CoT_prompt" "${json_cot_prompt}"
add_override "--datasets.vla_data.prefer_label_instruction" "${PREFER_LABEL_INSTRUCTION:-}"
add_override "--datasets.vla_data.per_device_batch_size" "${PER_DEVICE_BATCH_SIZE}"
add_override "--datasets.vla_data.num_workers" "${NUM_WORKERS}"
add_override "--datasets.vla_data.prefetch_factor" "${PREFETCH_FACTOR}"
add_override "--datasets.vla_data.skip_missing_datasets" "${SKIP_MISSING_DATASETS}"
add_override "--datasets.vla_data.include_state" "${INCLUDE_STATE}"
add_override "--framework.policy_context.use_state" "${INCLUDE_STATE}"
add_override "--datasets.vla_data.vlm_include_wrist_view" "${VLM_INCLUDE_WRIST_VIEW}"
add_override "--datasets.vla_data.wrist_history_frames" "${WRIST_HISTORY_FRAMES}"
add_override "--datasets.vla_data.future_wrist_frames" "${FUTURE_WRIST_FRAMES}"
add_override "--datasets.vla_data.load_wrist_future_views" "${LOAD_WRIST_FUTURE_VIEWS}"
add_override "--datasets.vla_data.future_wrist_k" "${FUTURE_WRIST_K}"
add_override "--datasets.vla_data.future_wrist_small_gap" "${FUTURE_WRIST_SMALL_GAP}"
add_override "--datasets.vla_data.action_type" "${ACTION_TYPE}"
add_override "--datasets.vla_data.action_mode" "${ACTION_MODE}"
add_override "--datasets.vla_data.video_backend" "${VIDEO_BACKEND}"

add_override "--framework.reasoning.cot_reasoning_mode" "${COT_REASONING_MODE}"
add_override "--framework.reasoning.latent_cot_num_queries" "${LATENT_COT_NUM_QUERIES}"
add_override "--framework.reasoning.qwen_action_context_mode" "${QWEN_ACTION_CONTEXT_MODE}"
add_override "--framework.reasoning.lambda_cot" "${LAMBDA_COT}"
add_override "--framework.reasoning.generate_cot_at_inference" "${GENERATE_COT_AT_INFERENCE}"

add_override "--framework.jepa_predictor.prediction_view" "${JEPA_PREDICTION_VIEW}"
add_override "--datasets.vla_data.jepa_prediction_view" "${JEPA_PREDICTION_VIEW}"
add_override "--framework.jepa_predictor.lambda_jepa" "${LAMBDA_JEPA}"
add_override "--framework.jepa_predictor.use_jepa_loss" "${USE_JEPA_LOSS}"
add_override "--framework.jepa_predictor.num_wrist_views" "${NUM_WRIST_VIEWS}"
add_override "--datasets.vla_data.num_wrist_views" "${NUM_WRIST_VIEWS}"
add_override "--framework.jepa_predictor.jepa_predictor_depth" "${JEPA_PREDICTOR_DEPTH}"
add_override "--framework.jepa_predictor.jepa_predictor_num_heads" "${JEPA_PREDICTOR_NUM_HEADS}"
add_override "--framework.jepa_predictor.jepa_predictor_activation_checkpointing" "${JEPA_PREDICTOR_ACTIVATION_CHECKPOINTING}"
add_override "--framework.policy_context.wrist_action_context_tokens" "${WRIST_ACTION_CONTEXT_TOKENS}"
add_override "--framework.action_model.gradient_checkpointing" "${ACTION_HEAD_GRADIENT_CHECKPOINTING}"

add_override "--framework.action_model.action_horizon" "${ACTION_HORIZON}"
add_override "--framework.action_model.future_action_window_size" "${FUTURE_ACTION_WINDOW_SIZE}"
add_override "--framework.action_model.past_action_window_size" "${PAST_ACTION_WINDOW_SIZE}"
add_override "--framework.action_model.action_dim" "${ACTION_DIM}"
add_override "--framework.action_model.state_dim" "${STATE_DIM}"

add_override "--trainer.max_train_steps" "${MAX_TRAIN_STEPS}"
add_override "--trainer.num_warmup_steps" "${NUM_WARMUP_STEPS}"
add_override "--trainer.gradient_accumulation_steps" "${GRADIENT_ACCUMULATION_STEPS}"
add_override "--trainer.repeated_diffusion_steps" "${REPEATED_DIFFUSION_STEPS}"
add_override "--framework.action_model.diffusion_model_cfg.dropout" "${DIT_DROPOUT}"
add_override "--trainer.train_qwen_vl" "${TRAIN_QWEN_VL}"
add_override "--trainer.find_unused_parameters" "${FIND_UNUSED_PARAMETERS}"
add_override "--trainer.is_resume" "${RESUME}"
add_override "--trainer.pretrained_checkpoint" "${PRETRAINED_CHECKPOINT}"
add_override "--trainer.reload_modules" "${RELOAD_MODULES}"
add_override "--trainer.resume_optimizer_state" "${RESUME_OPTIMIZER_STATE}"
add_override "--trainer.save_interval" "${SAVE_INTERVAL}"
add_override "--trainer.logging_frequency" "${LOGGING_FREQUENCY}"
add_override "--trainer.eval_interval" "${EVAL_INTERVAL}"
add_override "--run_root_dir" "${RUN_ROOT_DIR}"
add_override "--run_id" "${RUN_ID}"
add_override "--wandb_project" "${WANDB_PROJECT}"
add_override "--wandb_entity" "${WANDB_ENTITY}"

accelerate_args=(launch --num_processes "${NUM_PROCESSES}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}")
if [[ -n "${ACCELERATE_CONFIG}" ]]; then
  accelerate_args+=(--config_file "${ACCELERATE_CONFIG}")
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/accelerate" ]]; then
  accelerate_bin="${CONDA_PREFIX}/bin/accelerate"
else
  accelerate_bin="$(command -v accelerate)"
fi

echo "[W2 train] config=${CONFIG}"
echo "[W2 train] processes=${NUM_PROCESSES} deepspeed=${USE_DEEPSPEED} resume=${RESUME}"
echo "[W2 train] wandb_mode=${WANDB_MODE} wandb_key_set=$([[ -n "${WANDB_API_KEY}" ]] && printf yes || printf no)"

"${accelerate_bin}" "${accelerate_args[@]}" \
  starVLA/training/train_subtask_m2w_w2.py \
  --config_yaml "${CONFIG}" \
  "${override_args[@]}" \
  "$@"
