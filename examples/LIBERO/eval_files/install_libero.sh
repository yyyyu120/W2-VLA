#!/bin/bash
# Install LIBERO environment for evaluation
set -e

echo "=== Step 1: Activate libero conda env ==="
eval "$(conda shell.bash hook)"
conda activate libero

echo "=== Step 2: Install mujoco ==="
pip install mujoco

echo "=== Step 3: Clone and install LIBERO ==="
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/third_party}"
LIBERO_DIR="${LIBERO_HOME:-${INSTALL_ROOT}/LIBERO}"
if [ ! -d "$LIBERO_DIR" ]; then
    mkdir -p "${INSTALL_ROOT}"
    cd "${INSTALL_ROOT}"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd "${LIBERO_DIR}"
else
    echo "LIBERO already cloned at $LIBERO_DIR"
    cd "${LIBERO_DIR}"
fi

echo "=== Step 4: Install LIBERO (editable) ==="
pip install -e .

echo "=== Step 5: Install additional eval deps ==="
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4

echo "=== Step 6: Verify installation ==="
python -c "from libero.libero import benchmark; print('LIBERO OK:', benchmark)"
python -c "import mujoco; print('MuJoCo OK:', mujoco.__version__)"
python -c "import tyro; print('tyro OK')"
python -c "import websockets; print('websockets OK')"

echo "=== ALL DONE ==="
