# W2-VLA Training Pipeline

W2-VLA policy training consumes two separate inputs:

1. LeRobot action datasets containing images, robot states, and actions.
2. Frame-aligned offline CoT annotations from `W2-VLA-CoT`.

The source repository provides the training pipeline. The complete LeRobot
action datasets and prepared CoT annotations are published as separate
Hugging Face dataset repositories.

## Prepare Backbones

Download Qwen3-VL-4B-Instruct and V-JEPA2.1 as described in the main README.
V-JEPA2.1 is frozen during policy training.

## Prepare Data

Use this layout:

```text
playground/Datasets/
├── W2-VLA-Training-Data/
│   ├── libero/
│   ├── robotwin/
│   └── real_world/
└── W2-VLA-CoT/
    ├── libero/
    ├── robotwin/
    └── real_world/
```

Each CoT episode is named `episode_XXXXXX.npz`. Its `num_frames` value must
match the corresponding LeRobot episode length. The policy trains from the
`cot_train_text` field:

```text
Subtask: ...
Reasoning: ...
Wrist: ...
```

See [`../docs/DATASETS.md`](../docs/DATASETS.md) for download commands, release
statistics, and the complete schema.

## Train

Use `scripts/train_subtask_m2w_deepspeed.sh` with the matching configuration:

```text
LIBERO:   starVLA/config/training/subtask_m2w_libero_w2.yaml
RoboTwin: starVLA/config/training/subtask_m2w_robotwin_w2.yaml
```

Real-world tasks use the task-specific wrappers under `scripts/`. All wrappers
accept `DATA_ROOT_DIR`, `SUBTASK_LABEL_DIR`, and `BASE_VLM` overrides.

## Evaluate

Evaluation requires a policy checkpoint together with the exact `config.yaml`
and `dataset_statistics.json` produced by its training run. Offline CoT files
are not loaded during inference.
