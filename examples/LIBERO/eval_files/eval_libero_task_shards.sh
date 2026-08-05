#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${STARVLA_DIR}"

CKPT="${CKPT:?Please set CKPT=/path/to/steps_xxx_pytorch_model.pt}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-20}"
NUM_TASKS="${NUM_TASKS:-10}"
GPU_LIST_RAW="${GPU_LIST:-0,1,2,3,4}"
BASE_PORT="${BASE_PORT:-6694}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
SERVER_STAGGER_SECONDS="${SERVER_STAGGER_SECONDS:-8}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_Python:-python}}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"

IFS=', ' read -r -a GPU_LIST <<< "${GPU_LIST_RAW}"
NUM_SHARDS="${#GPU_LIST[@]}"
if [[ "${NUM_SHARDS}" -le 0 ]]; then
    echo "[ERROR] Empty GPU_LIST=${GPU_LIST_RAW}" >&2
    exit 1
fi

folder_name=$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
LOG_ROOT="${LOG_ROOT:-results/eval_shards/${TASK_SUITE_NAME}/${folder_name}_$(date +"%Y%m%d_%H%M%S")}"
mkdir -p "${LOG_ROOT}"

pids=()
server_pids=()
server_logs=()

cleanup() {
    for pid in "${server_pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

wait_for_server_log() {
    local log_path="$1"
    local server_pid="$2"
    local port="$3"
    local timeout="$4"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if [[ -f "${log_path}" ]] && grep -q "server listening on .*:${port}" "${log_path}"; then
            return 0
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            return 1
        fi
        if (( $(date +%s) - start_ts > timeout )); then
            return 1
        fi
        sleep 2
    done
}

echo "=========================================="
echo " LIBERO task-sharded eval"
echo " ckpt        : ${CKPT}"
echo " suite       : ${TASK_SUITE_NAME}"
echo " num_tasks   : ${NUM_TASKS}"
echo " trials/task : ${NUM_TRIALS_PER_TASK}"
echo " gpu_list    : ${GPU_LIST[*]}"
echo " base_port   : ${BASE_PORT}"
echo " save_video  : ${SAVE_VIDEO}"
echo " log_root    : ${LOG_ROOT}"
echo "=========================================="

for ((i=0; i<NUM_SHARDS; i++)); do
    gpu_id="${GPU_LIST[$i]}"
    port=$((BASE_PORT + i))
    task_start=$(( i * NUM_TASKS / NUM_SHARDS ))
    task_end=$(( (i + 1) * NUM_TASKS / NUM_SHARDS ))
    if [[ "${task_start}" -ge "${task_end}" ]]; then
        echo "[Shard ${i}] skip empty range [${task_start}, ${task_end})"
        continue
    fi

    server_log="${LOG_ROOT}/server_gpu${gpu_id}_port${port}_task${task_start}_${task_end}.log"
    echo "[Shard ${i}] start server GPU=${gpu_id} port=${port} tasks=[${task_start}, ${task_end})"
    GPU_ID="${gpu_id}" \
    PORT="${port}" \
    CKPT="${CKPT}" \
    STARVLA_PYTHON="${STARVLA_PYTHON}" \
    bash examples/LIBERO/eval_files/run_policy_server.sh >"${server_log}" 2>&1 &
    server_pids[$i]="$!"
    server_logs[$i]="${server_log}"
    sleep "${SERVER_STAGGER_SECONDS}"
done

for ((i=0; i<NUM_SHARDS; i++)); do
    gpu_id="${GPU_LIST[$i]}"
    port=$((BASE_PORT + i))
    task_start=$(( i * NUM_TASKS / NUM_SHARDS ))
    task_end=$(( (i + 1) * NUM_TASKS / NUM_SHARDS ))
    if [[ "${task_start}" -ge "${task_end}" ]]; then
        continue
    fi
    echo "[Shard ${i}] wait for server port=${port}"
    if ! wait_for_server_log "${server_logs[$i]}" "${server_pids[$i]}" "${port}" "${SERVER_READY_TIMEOUT}"; then
        echo "[ERROR] Server on port ${port} did not become ready. See ${LOG_ROOT}/server_gpu${gpu_id}_port${port}_task${task_start}_${task_end}.log" >&2
        exit 1
    fi
done

for ((i=0; i<NUM_SHARDS; i++)); do
    gpu_id="${GPU_LIST[$i]}"
    port=$((BASE_PORT + i))
    task_start=$(( i * NUM_TASKS / NUM_SHARDS ))
    task_end=$(( (i + 1) * NUM_TASKS / NUM_SHARDS ))
    if [[ "${task_start}" -ge "${task_end}" ]]; then
        continue
    fi

    eval_log="${LOG_ROOT}/eval_gpu${gpu_id}_port${port}_task${task_start}_${task_end}.log"
    echo "[Shard ${i}] eval GPU=${gpu_id} port=${port} tasks=[${task_start}, ${task_end})"
    LIBERO_PYTHON="${LIBERO_PYTHON}" \
    PORT="${port}" \
    TASK_SUITE_NAME="${TASK_SUITE_NAME}" \
    TASK_ID_START="${task_start}" \
    TASK_ID_END="${task_end}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
    SAVE_VIDEO="${SAVE_VIDEO}" \
    CKPT="${CKPT}" \
    bash examples/LIBERO/eval_files/eval_libero.sh >"${eval_log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

echo "=========================================="
echo " Sharded eval finished with status=${status}"
echo " Logs: ${LOG_ROOT}"
echo " Per-shard success can be read from eval_*.log"
echo "=========================================="
exit "${status}"
