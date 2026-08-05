# LIBERO

This directory contains the LIBERO data preparation, training, and evaluation
entrypoints used by the project.

## Environment

Create one environment for policy training and serving, and a separate LIBERO
environment for simulation. Install the project dependencies in the policy
environment and follow the official LIBERO installation instructions in the
simulation environment.

## Data

Download the prepared action-data release as described in the main README. The
four suites are directly available under:

```text
playground/Datasets/W2-VLA-Training-Data/libero/
```

Alternatively, convert LIBERO independently with `data_preparation.sh` and
pass the resulting root through `DATA_ROOT_DIR`.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=16 \
DATA_ROOT_DIR=$PWD/playground/Datasets/W2-VLA-Training-Data/libero \
SUBTASK_LABEL_DIR=$PWD/playground/Datasets/W2-VLA-CoT/libero \
BASE_VLM=$PWD/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
RUN_ID=libero_training \
WANDB_MODE=online \
bash scripts/train_subtask_m2w_deepspeed.sh \
  starVLA/config/training/subtask_m2w_libero_w2.yaml
```

For a smoke test, reduce `PER_DEVICE_BATCH_SIZE` and `MAX_TRAIN_STEPS`.

## Evaluation

Start the policy server in the policy environment:

```bash
GPU_ID=0 \
PORT=6694 \
CKPT=/path/to/checkpoint.pt \
bash examples/LIBERO/eval_files/run_policy_server.sh
```

Run the simulator client in the LIBERO environment:

```bash
LIBERO_HOME=/path/to/LIBERO \
PORT=6694 \
TASK_SUITE_NAME=libero_10 \
NUM_TRIALS_PER_TASK=20 \
SAVE_VIDEO=0 \
CKPT=/path/to/checkpoint.pt \
bash examples/LIBERO/eval_files/eval_libero.sh
```

Use `eval_libero_task_shards.sh` for task-level multi-GPU evaluation.
