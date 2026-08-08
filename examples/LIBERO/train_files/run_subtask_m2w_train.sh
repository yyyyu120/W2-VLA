#!/bin/bash

set -euo pipefail

export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE=${WANDB_MODE:-offline}
use_deepspeed=${use_deepspeed:-false}

if [[ "${use_deepspeed}" == "true" || "${use_deepspeed}" == "1" ]]; then
  export STARVLA_USE_DEEPSPEED=1
  # DeepSpeed import/build checks need CUDA_HOME/bin/nvcc on some conda setups.
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
else
  export STARVLA_USE_DEEPSPEED=0
fi

###########################################################################################
# === Please modify the following paths according to your environment ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STARVLA_DIR=${STARVLA_DIR:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}

base_vlm=${base_vlm:-${STARVLA_DIR}/playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-${STARVLA_DIR}/starVLA/config/training/subtask_m2w_libero.yaml}
libero_data_root=${libero_data_root:-${STARVLA_DIR}/playground/Datasets/LEROBOT_LIBERO_DATA}
subtask_label_dir=${subtask_label_dir:-${STARVLA_DIR}/playground/Datasets/W2-VLA-CoT/libero}
cot_prompt_version=${cot_prompt_version:-}
vjepa_backend=${vjepa_backend:-torchhub}
vjepa_base_encoder=${vjepa_base_encoder:-facebook/vjepa2-vitl-fpc64-256}
vjepa_image_size=${vjepa_image_size:-}
vjepa_num_frames=${vjepa_num_frames:-8}
vjepa_checkpoint=${vjepa_checkpoint:-${STARVLA_DIR}/playground/Pretrained_models/VJEPA2.1/vjepa2_1_vitl_dist_vitG_384.pt}
vjepa_hub_repo=${vjepa_hub_repo:-facebookresearch/vjepa2:main}
vjepa_hub_source=${vjepa_hub_source:-github}
data_mix=${data_mix:-libero_all}
include_state=${include_state:-false}
run_root_dir=${run_root_dir:-${STARVLA_DIR}/results/Checkpoints}
run_id=${run_id:-subtask_m2w_libero}
num_processes=${num_processes:-4}
per_device_batch_size=${per_device_batch_size:-16}
num_workers=${num_workers:-2}
prefetch_factor=${prefetch_factor:-2}
future_wrist_target=${future_wrist_target:-}
wrist_history_frames=${wrist_history_frames:-}
load_wrist_future_views=${load_wrist_future_views:-}
gradient_accumulation_steps=${gradient_accumulation_steps:-1}
export STARVLA_GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}"
max_train_steps=${max_train_steps:-80000}
num_warmup_steps=${num_warmup_steps:-}
repeated_diffusion_steps=${repeated_diffusion_steps:-8}
dit_dropout=${dit_dropout:-}
wandb_project=${wandb_project:-w2-vla}
wandb_entity=${wandb_entity:-}
skip_vjepa_check=${skip_vjepa_check:-0}
skip_vjepa_init=${skip_vjepa_init:-}
train_qwen_vl=${train_qwen_vl:-true}
find_unused_parameters=${find_unused_parameters:-false}
action_condition_mode=${action_condition_mode:-}
query_condition_mode=${query_condition_mode:-}
cot_reasoning_mode=${cot_reasoning_mode:-}
latent_cot_num_queries=${latent_cot_num_queries:-}
latent_cot_max_text_tokens=${latent_cot_max_text_tokens:-}
future_predictor_type=${future_predictor_type:-}
future_predictor_bottleneck_dim=${future_predictor_bottleneck_dim:-}
future_predictor_num_layers=${future_predictor_num_layers:-}
future_predictor_num_heads=${future_predictor_num_heads:-}
use_ema_target_projector=${use_ema_target_projector:-}
target_ema_decay=${target_ema_decay:-}
qwen_input_views=${qwen_input_views:-}
qwen_action_context_mode=${qwen_action_context_mode:-}
use_cot_loss=${use_cot_loss:-}
use_wrist_future_loss=${use_wrist_future_loss:-}
include_wrist_query_in_action=${include_wrist_query_in_action:-}
include_future_wrist_in_action=${include_future_wrist_in_action:-}
lambda_cot=${lambda_cot:-}
lambda_wrist=${lambda_wrist:-}
generate_cot_at_inference=${generate_cot_at_inference:-}
adapter_lr=${adapter_lr:-5.0e-5}
action_lr=${action_lr:-1.0e-4}
vlm_lr=${vlm_lr:-1.0e-5}
pretrained_checkpoint=${pretrained_checkpoint:-}
is_resume=${is_resume:-false}
save_optimizer_state=${save_optimizer_state:-false}
resume_optimizer_state=${resume_optimizer_state:-true}
# === End of environment variable configuration ===
###########################################################################################

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
if [[ -z "${vjepa_image_size}" ]]; then
  if [[ "${vjepa_backend}" == "hf" ]]; then
    vjepa_image_size=256
  else
    vjepa_image_size=384
  fi
fi

required_files=(
  "starVLA/training/train_subtask_m2w.py"
  "starVLA/model/framework/QwenSubtaskM2W.py"
  "starVLA/model/modules/main_to_wrist/main_to_wrist_adapter.py"
  "starVLA/model/modules/frozen_visual_encoder/vjepa2_encoder.py"
  "starVLA/dataloader/subtask_m2w_datasets.py"
  "starVLA/config/training/subtask_m2w_libero.yaml"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[M2W] Missing required file: ${STARVLA_DIR}/${required_file}" >&2
    echo "[M2W] Please sync the M2W code files before training." >&2
    exit 1
  fi
done
if [[ ! -d "${subtask_label_dir}" ]]; then
  echo "[M2W] Missing subtask label dir: ${subtask_label_dir}" >&2
  exit 1
fi
skip_vjepa_enabled=false
if [[ "${skip_vjepa_init}" == "true" || "${skip_vjepa_init}" == "1" ]]; then
  skip_vjepa_enabled=true
fi

if [[ "${skip_vjepa_enabled}" == "true" ]]; then
  echo "[M2W] skip_vjepa_init=true: skipping V-JEPA checkpoint/link/preload checks."
elif [[ "${vjepa_backend}" == "torchhub" && -f "${vjepa_checkpoint}" ]]; then
  torchhub_ckpt_dir="$(python -c 'import torch; print(torch.hub.get_dir() + "/checkpoints")')"
  mkdir -p "${torchhub_ckpt_dir}"
  ln -sf "${vjepa_checkpoint}" "${torchhub_ckpt_dir}/$(basename "${vjepa_checkpoint}")"
  echo "[M2W] Linked V-JEPA checkpoint into torchhub cache:"
  echo "[M2W]   ${torchhub_ckpt_dir}/$(basename "${vjepa_checkpoint}") -> ${vjepa_checkpoint}"
elif [[ "${vjepa_backend}" == "torchhub" ]]; then
  echo "[M2W] V-JEPA checkpoint not found: ${vjepa_checkpoint}" >&2
  echo "[M2W] Set vjepa_checkpoint=/path/to/vjepa2_1_vitl_dist_vitG_384.pt or run scripts/download_vjepa2_1_checkpoint.sh." >&2
  exit 1
fi
if [[ "${skip_vjepa_enabled}" != "true" && "${vjepa_hub_source}" == "local" && ! -f "${vjepa_hub_repo}/hubconf.py" ]]; then
  vjepa_hubconf="$(find "${vjepa_hub_repo}" -maxdepth 3 -name hubconf.py -print -quit 2>/dev/null || true)"
  if [[ -n "${vjepa_hubconf}" ]]; then
    vjepa_hub_repo="$(cd "$(dirname "${vjepa_hubconf}")" && pwd)"
    echo "[M2W] Resolved local V-JEPA hub repo to: ${vjepa_hub_repo}"
  else
    echo "[M2W] Local V-JEPA repo is missing hubconf.py: ${vjepa_hub_repo}" >&2
    echo "[M2W] vjepa_hub_repo must point to the root of facebookresearch/vjepa2, not only a checkpoint directory." >&2
    echo "[M2W] Example:" >&2
    echo "[M2W]   git clone https://github.com/facebookresearch/vjepa2.git ${STARVLA_DIR}/playground/Pretrained_models/vjepa2" >&2
    exit 1
  fi
fi

python -c 'import os, starVLA; root=os.path.realpath(os.environ["STARVLA_DIR"]); path=os.path.realpath(starVLA.__file__); print(f"[M2W] starVLA import path: {path}"); assert path.startswith(root + os.sep), f"starVLA is imported from {path}, expected under {root}. Check PYTHONPATH or editable installs."'

if [[ "${skip_vjepa_enabled}" != "true" && "${skip_vjepa_check}" != "1" ]]; then
  echo "[M2W] Preloading V-JEPA2.1 once before distributed training..."
  vjepa_check_args=(
    scripts/check_vjepa2_encoder.py
    --backend "${vjepa_backend}" \
    --image-size "${vjepa_image_size}" \
    --num-frames "${vjepa_num_frames}" \
    --batch-size 1
  )
  if [[ "${vjepa_backend}" == "hf" ]]; then
    vjepa_check_args+=(--base-encoder "${vjepa_base_encoder}")
  else
    vjepa_check_args+=(
      --hub-repo "${vjepa_hub_repo}"
      --hub-model vjepa2_1_vit_large_384
      --hub-source "${vjepa_hub_source}"
    )
  fi
  if python scripts/check_vjepa2_encoder.py --help | grep -q -- "--max-tokens"; then
    vjepa_check_args+=(--max-tokens 256)
  fi
  python "${vjepa_check_args[@]}"
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "${SCRIPT_DIR}/run_subtask_m2w_train.sh" "${output_dir}/"

vjepa_train_args=()
if [[ "${skip_vjepa_enabled}" != "true" ]]; then
  vjepa_train_args=(
    --framework.vjepa2.backend "${vjepa_backend}"
    --framework.vjepa2.image_size "${vjepa_image_size}"
    --framework.vjepa2.num_frames "${vjepa_num_frames}"
  )
  if [[ "${vjepa_backend}" == "hf" ]]; then
    vjepa_train_args+=(--framework.vjepa2.base_encoder "${vjepa_base_encoder}")
  else
    vjepa_train_args+=(
      --framework.vjepa2.hub_repo "${vjepa_hub_repo}"
      --framework.vjepa2.hub_source "${vjepa_hub_source}"
    )
  fi
fi

pretrained_args=()
if [[ -n "${pretrained_checkpoint}" ]]; then
  pretrained_args+=(--trainer.pretrained_checkpoint "${pretrained_checkpoint}")
fi

cot_prompt_args=()
if [[ -n "${cot_prompt_version}" ]]; then
  # Clear the YAML prompt so QwenSubtaskM2W resolves the requested immutable
  # prompt version, then saves the resulting full text into the run config.
  cot_prompt_args+=(
    --datasets.vla_data.cot_prompt_version "${cot_prompt_version}"
    --datasets.vla_data.CoT_prompt ""
  )
fi

dataset_override_args=()
if [[ -n "${include_state}" ]]; then
  dataset_override_args+=(--datasets.vla_data.include_state "${include_state}")
fi
if [[ -n "${future_wrist_target}" ]]; then
  dataset_override_args+=(--datasets.vla_data.future_wrist_target "${future_wrist_target}")
fi
if [[ -n "${wrist_history_frames}" ]]; then
  dataset_override_args+=(--datasets.vla_data.wrist_history_frames "${wrist_history_frames}")
fi
if [[ -n "${load_wrist_future_views}" ]]; then
  dataset_override_args+=(--datasets.vla_data.load_wrist_future_views "${load_wrist_future_views}")
fi

m2w_ablation_args=()
if [[ -n "${action_condition_mode}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.action_condition_mode "${action_condition_mode}")
fi
if [[ -n "${query_condition_mode}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.query_condition_mode "${query_condition_mode}")
fi
if [[ -n "${cot_reasoning_mode}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.cot_reasoning_mode "${cot_reasoning_mode}")
fi
if [[ -n "${latent_cot_num_queries}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.latent_cot_num_queries "${latent_cot_num_queries}")
fi
if [[ -n "${latent_cot_max_text_tokens}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.latent_cot_max_text_tokens "${latent_cot_max_text_tokens}")
fi
if [[ -n "${future_predictor_type}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.future_predictor_type "${future_predictor_type}")
fi
if [[ -n "${future_predictor_bottleneck_dim}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.future_predictor_bottleneck_dim "${future_predictor_bottleneck_dim}")
fi
if [[ -n "${future_predictor_num_layers}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.future_predictor_num_layers "${future_predictor_num_layers}")
fi
if [[ -n "${future_predictor_num_heads}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.future_predictor_num_heads "${future_predictor_num_heads}")
fi
if [[ -n "${use_ema_target_projector}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.use_ema_target_projector "${use_ema_target_projector}")
fi
if [[ -n "${target_ema_decay}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.target_ema_decay "${target_ema_decay}")
fi
if [[ -n "${qwen_input_views}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.qwen_input_views "${qwen_input_views}")
fi
if [[ -n "${qwen_action_context_mode}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.qwen_action_context_mode "${qwen_action_context_mode}")
fi
if [[ -n "${use_cot_loss}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.use_cot_loss "${use_cot_loss}")
fi
if [[ -n "${use_wrist_future_loss}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.use_wrist_future_loss "${use_wrist_future_loss}")
fi
if [[ -n "${skip_vjepa_init}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.skip_vjepa_init "${skip_vjepa_init}")
fi
if [[ -n "${include_wrist_query_in_action}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.include_wrist_query_in_action "${include_wrist_query_in_action}")
fi
if [[ -n "${include_future_wrist_in_action}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.include_future_wrist_in_action "${include_future_wrist_in_action}")
fi
if [[ -n "${lambda_cot}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.lambda_cot "${lambda_cot}")
fi
if [[ -n "${lambda_wrist}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.lambda_wrist "${lambda_wrist}")
fi
if [[ -n "${generate_cot_at_inference}" ]]; then
  m2w_ablation_args+=(--framework.main_to_wrist.generate_cot_at_inference "${generate_cot_at_inference}")
fi

training_override_args=()
if [[ -n "${num_warmup_steps}" ]]; then
  training_override_args+=(--trainer.num_warmup_steps "${num_warmup_steps}")
fi
if [[ -n "${repeated_diffusion_steps}" ]]; then
  # QwenSubtaskM2W reads this trainer value when repeating each action target
  # over independently sampled flow-matching noise levels. Keep the action
  # model copy in sync so the saved run config is self-describing.
  training_override_args+=(
    --trainer.repeated_diffusion_steps "${repeated_diffusion_steps}"
    --framework.action_model.repeated_diffusion_steps "${repeated_diffusion_steps}"
  )
fi
if [[ -n "${dit_dropout}" ]]; then
  training_override_args+=(--framework.action_model.diffusion_model_cfg.dropout "${dit_dropout}")
fi

accelerate_launch_args=(
  --num_processes="${num_processes}"
  --num_machines=1
  --dynamo_backend=no
  --mixed_precision=bf16
)
if [[ "${use_deepspeed}" == "true" || "${use_deepspeed}" == "1" ]]; then
  accelerate_launch_args=(
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
    --num_processes "${num_processes}"
    --gradient_accumulation_steps "${gradient_accumulation_steps}"
  )
fi

accelerate launch \
  "${accelerate_launch_args[@]}" \
  starVLA/training/train_subtask_m2w.py \
  --config_yaml "${config_yaml}" \
  --framework.name QwenSubtaskM2W \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  "${vjepa_train_args[@]}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.subtask_label_dir "${subtask_label_dir}" \
  "${cot_prompt_args[@]}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.num_workers "${num_workers}" \
  --datasets.vla_data.prefetch_factor "${prefetch_factor}" \
  "${dataset_override_args[@]}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  "${training_override_args[@]}" \
  --trainer.train_qwen_vl "${train_qwen_vl}" \
  --trainer.find_unused_parameters "${find_unused_parameters}" \
  --trainer.is_resume "${is_resume}" \
  --trainer.learning_rate.adapter "${adapter_lr}" \
  --trainer.learning_rate.action_model "${action_lr}" \
  --trainer.learning_rate.vlm "${vlm_lr}" \
  --trainer.save_optimizer_state "${save_optimizer_state}" \
  --trainer.resume_optimizer_state "${resume_optimizer_state}" \
  "${pretrained_args[@]}" \
  "${m2w_ablation_args[@]}" \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  2>&1 | tee "${output_dir}/train.log"
