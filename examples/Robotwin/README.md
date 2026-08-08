# RoboTwin

This directory contains the RoboTwin training and evaluation entrypoints used
by the project.

## Environment

Use separate environments for the policy server and the RoboTwin simulator.
Install the project dependencies in the policy environment and follow the
official RoboTwin installation guide for the simulator environment.

Set the paths before evaluation:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin-Platform
export STARVLA_PYTHON=/path/to/policy-env/bin/python
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python
export PYTHONNOUSERSITE=1
```

## Data

Place the LeRobot-formatted tasks under:

```text
playground/Datasets/W2-VLA-Training-Data/robotwin/<task-name>/
```

Store the matching labels under `playground/Datasets/W2-VLA-CoT/robotwin`.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=1 \
DATA_ROOT_DIR=$PWD/playground/Datasets/W2-VLA-Training-Data/robotwin \
SUBTASK_LABEL_DIR=$PWD/playground/Datasets/W2-VLA-CoT/robotwin \
BASE_VLM=$PWD/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
RUN_ID=robotwin_h16_training \
WANDB_MODE=online \
bash scripts/train_subtask_m2w_deepspeed.sh \
  starVLA/config/training/subtask_m2w_robotwin_w2.yaml
```

The default configuration uses a 16-step absolute-action chunk, two wrist
views, and eight historical wrist frames sampled with stride two.

## Evaluation

Evaluate one task with the unified launcher:

```bash
ROBOTWIN_TEST_NUM=100 \
ROBOTWIN_EVAL_VIDEO_LOG=False \
ROBOTWIN_FLOW_INFERENCE_STEPS=8 \
CUDA_VISIBLE_DEVICES=0 \
bash examples/Robotwin/eval_files/start_eval.sh \
  -m demo_clean \
  -n robotwin_evaluation \
  -c /path/to/checkpoint.pt \
  -s 0 \
  -j 1 \
  -p 6794 \
  adjust_bottle
```

Supported scene modes are `demo_clean` and `demo_randomized`. Multiple tasks
may be listed after the options, and `CUDA_VISIBLE_DEVICES` may contain several
GPUs. The launcher writes separate policy-server and simulator logs for each
task and cleans up its own child processes when it exits.
