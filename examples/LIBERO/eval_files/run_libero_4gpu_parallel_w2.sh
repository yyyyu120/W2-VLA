#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${STARVLA_DIR}"

CKPT="${CKPT:?Set CKPT=/path/to/steps_N_pytorch_model.pt}"
SUITES_STR="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
GPUS_STR="${GPUS:-0 1 2 3}"
BASE_PORT="${BASE_PORT:-6694}"
PORTS_STR="${PORTS:-}"
SERVER_START_DELAY="${SERVER_START_DELAY:-5}"
LOG_ROOT="${LOG_ROOT:-logs/libero_4gpu_$(date +"%Y%m%d_%H%M%S")}"

read -r -a SUITES_ARR <<< "${SUITES_STR}"
read -r -a GPUS_ARR <<< "${GPUS_STR}"

if [ -n "${PORTS_STR}" ]; then
    read -r -a PORTS_ARR <<< "${PORTS_STR}"
else
    PORTS_ARR=()
    for idx in "${!SUITES_ARR[@]}"; do
        PORTS_ARR+=("$((BASE_PORT + idx))")
    done
fi

if [ "${#SUITES_ARR[@]}" -eq 0 ]; then
    echo "[ERROR] SUITES is empty." >&2
    exit 1
fi

if [ "${#SUITES_ARR[@]}" -ne "${#GPUS_ARR[@]}" ]; then
    echo "[ERROR] SUITES count (${#SUITES_ARR[@]}) must match GPUS count (${#GPUS_ARR[@]})." >&2
    exit 1
fi

if [ "${#SUITES_ARR[@]}" -ne "${#PORTS_ARR[@]}" ]; then
    echo "[ERROR] SUITES count (${#SUITES_ARR[@]}) must match PORTS count (${#PORTS_ARR[@]})." >&2
    exit 1
fi

for i in "${!PORTS_ARR[@]}"; do
    for j in "${!PORTS_ARR[@]}"; do
        if [ "${i}" -lt "${j}" ] && [ "${PORTS_ARR[$i]}" = "${PORTS_ARR[$j]}" ]; then
            echo "[ERROR] Duplicate port configured: ${PORTS_ARR[$i]}" >&2
            exit 1
        fi
    done
done

if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
    exit 1
fi

for port in "${PORTS_ARR[@]}"; do
    if bash -c ":</dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
        echo "[ERROR] Port ${port} is already in use. Set BASE_PORT or PORTS to free ports." >&2
        exit 1
    fi
done

mkdir -p "${LOG_ROOT}"

job_pids=()

cleanup_all() {
    echo "[INFO] Cleaning up background jobs and policy servers..."
    for pid in "${job_pids[@]:-}"; do
        if kill -0 "${pid}" >/dev/null 2>&1; then
            kill "${pid}" >/dev/null 2>&1 || true
        fi
    done
    if [ -d "${LOG_ROOT}" ]; then
        while IFS= read -r pid_file; do
            server_pid="$(cat "${pid_file}" 2>/dev/null || true)"
            if [ -n "${server_pid}" ] && kill -0 "${server_pid}" >/dev/null 2>&1; then
                kill "${server_pid}" >/dev/null 2>&1 || true
            fi
        done < <(find "${LOG_ROOT}" -name server.pid -type f 2>/dev/null)
    fi
}

trap 'cleanup_all; exit 130' INT
trap 'cleanup_all; exit 143' TERM

run_one_suite() {
    set +e

    local suite="$1"
    local gpu="$2"
    local port="$3"
    local log_dir="${LOG_ROOT}/${suite}"
    local server_log="${log_dir}/server.log"
    local eval_log="${log_dir}/eval.log"
    local server_pid
    local server_status
    local eval_status

    mkdir -p "${log_dir}"

    {
        echo "suite=${suite}"
        echo "gpu=${gpu}"
        echo "port=${port}"
        echo "ckpt=${CKPT}"
        echo "started_at=$(date +"%Y-%m-%d %H:%M:%S")"
    } > "${log_dir}/run.env"

    echo "[${suite}] starting policy server on GPU ${gpu}, port ${port}"
    GPU_ID="${gpu}" PORT="${port}" CKPT="${CKPT}" \
        bash "${SCRIPT_DIR}/run_policy_server_w2.sh" > "${server_log}" 2>&1 &
    server_pid=$!
    echo "${server_pid}" > "${log_dir}/server.pid"

    sleep "${SERVER_START_DELAY}"
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
        wait "${server_pid}" >/dev/null 2>&1
        server_status=$?
        if [ "${server_status}" -eq 0 ]; then
            server_status=1
        fi
        echo "${server_status}" > "${log_dir}/exit_code"
        echo "[${suite}] policy server exited before eval started. See ${server_log}"
        return "${server_status}"
    fi

    echo "[${suite}] starting eval client on GPU ${gpu}, port ${port}"
    CUDA_VISIBLE_DEVICES="${gpu}" PORT="${port}" TASK_SUITE_NAME="${suite}" CKPT="${CKPT}" \
        bash "${SCRIPT_DIR}/eval_libero_w2.sh" > "${eval_log}" 2>&1
    eval_status=$?

    if kill -0 "${server_pid}" >/dev/null 2>&1; then
        kill "${server_pid}" >/dev/null 2>&1 || true
        wait "${server_pid}" >/dev/null 2>&1 || true
    fi

    echo "${eval_status}" > "${log_dir}/exit_code"
    if [ "${eval_status}" -eq 0 ]; then
        echo "[${suite}] completed successfully"
    else
        echo "[${suite}] failed with exit code ${eval_status}. See ${eval_log}"
    fi
    return "${eval_status}"
}

echo "=========================================="
echo " LIBERO 4-GPU Parallel Eval"
echo "=========================================="
echo " Checkpoint : ${CKPT}"
echo " Suites     : ${SUITES_ARR[*]}"
echo " GPUs       : ${GPUS_ARR[*]}"
echo " Ports      : ${PORTS_ARR[*]}"
echo " Logs       : ${LOG_ROOT}"
echo "=========================================="

for idx in "${!SUITES_ARR[@]}"; do
    run_one_suite "${SUITES_ARR[$idx]}" "${GPUS_ARR[$idx]}" "${PORTS_ARR[$idx]}" &
    job_pids+=("$!")
    sleep 2
done

overall_status=0
for idx in "${!job_pids[@]}"; do
    suite="${SUITES_ARR[$idx]}"
    pid="${job_pids[$idx]}"
    if wait "${pid}"; then
        echo "[INFO] ${suite}: OK"
    else
        status=$?
        echo "[ERROR] ${suite}: failed with exit code ${status}"
        overall_status=1
    fi
done

cleanup_all

echo "=========================================="
if [ "${overall_status}" -eq 0 ]; then
    echo " All LIBERO evaluations completed successfully."
else
    echo " Some LIBERO evaluations failed. Check logs under ${LOG_ROOT}."
fi
echo "=========================================="

exit "${overall_status}"
