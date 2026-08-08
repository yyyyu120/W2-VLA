#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${STARVLA_DIR}"
CKPT="${CKPT:?Set CKPT=/path/to/steps_N_pytorch_model.pt}"


###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/third_party/LIBERO}"
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_Python:-python}}"

export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}" # let eval_libero find the LIBERO tools
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH}" # let LIBERO find the websocket tools from main repo

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl


host="${HOST:-127.0.0.1}"
base_port="${PORT:-6694}"
unnorm_key="franka"
your_ckpt=${CKPT}


# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

task_suite_name="${TASK_SUITE_NAME:-libero_goal}"
task_id_start="${TASK_ID_START:-0}"
task_id_end="${TASK_ID_END:--1}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-20}"
save_video="${SAVE_VIDEO:-1}"
if [[ "${task_id_start}" != "0" || "${task_id_end}" != "-1" ]]; then
    video_out_path="results/${task_suite_name}/${folder_name}/task_${task_id_start}_${task_id_end}"
else
    video_out_path="results/${task_suite_name}/${folder_name}"
fi

extra_args=()
extra_args+=(--args.task-id-start "${task_id_start}")
extra_args+=(--args.task-id-end "${task_id_end}")
if [[ "${save_video}" == "0" || "${save_video}" == "false" || "${save_video}" == "False" ]]; then
    extra_args+=(--args.no-save-video)
fi


${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    "${extra_args[@]}"
