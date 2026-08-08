#!/usr/bin/env bash
# Portable RoboTwin evaluation environment.
# Usage:
#   source scripts/env_robotwin.sh
#   ROBOTWIN_PATH=/path/to/RoboTwin-Platform \
#   ROBOTWIN_TEST_NUM=100 CUDA_VISIBLE_DEVICES=0 \
#   bash examples/Robotwin/eval_files/start_eval.sh ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export M2W_VLA_ROOT=${M2W_VLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}

# Set this explicitly when RoboTwin is not checked out next to this repository.
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-$(cd "${M2W_VLA_ROOT}/.." && pwd)/RoboTwin-Platform}

# Conda environment names. These are used by start_eval.sh if explicit python paths are absent.
export ROBOTWIN_STARVLA_ENV=${ROBOTWIN_STARVLA_ENV:-starVLA}
export ROBOTWIN_ENV=${ROBOTWIN_ENV:-robotwin}

# Explicit interpreters are optional. The active interpreter is used by
# default; override these when policy and simulator run in separate envs.
export STARVLA_PYTHON=${STARVLA_PYTHON:-$(command -v python)}
export ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-$(command -v python)}

# Avoid accidental user-site package leakage between the two environments.
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

# Default evaluation behavior. Override per command when needed.
export ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-False}
export ROBOTWIN_TEST_NUM=${ROBOTWIN_TEST_NUM:-100}

# Keep imports predictable for scripts launched from arbitrary cwd.
export PYTHONPATH=${M2W_VLA_ROOT}:${PYTHONPATH:-}

# Optional mirror; leave unset by default for eval. Uncomment if a script downloads from HF.
# export HF_ENDPOINT=https://hf-mirror.com

if [[ ! -d "${M2W_VLA_ROOT}" ]]; then
  echo "[env_robotwin] M2W_VLA_ROOT does not exist: ${M2W_VLA_ROOT}" >&2
fi
if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
  echo "[env_robotwin] ROBOTWIN_PATH does not exist: ${ROBOTWIN_PATH}" >&2
fi
if [[ ! -x "${STARVLA_PYTHON}" ]]; then
  echo "[env_robotwin] STARVLA_PYTHON not executable: ${STARVLA_PYTHON}" >&2
fi
if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
  echo "[env_robotwin] ROBOTWIN_PYTHON not executable: ${ROBOTWIN_PYTHON}" >&2
fi
