# Data and Checkpoints

W²-VLA keeps datasets, offline CoT labels, pretrained backbones, and training
run artifacts under `playground/` so that source code and large assets remain
cleanly separated.

## Local Layout

The default scripts expect external assets under `playground/`:

```text
playground/
  Datasets/
    W2-VLA-Training-Data/
      libero/
      robotwin/
      real_world/
    W2-VLA-CoT/
      libero/
      robotwin/
      real_world/
  Pretrained_models/
    Qwen3-VL-4B-Instruct/
    VJEPA2.1/
```

These paths are ignored by Git. They can be replaced at launch time:

```bash
DATA_ROOT_DIR=/path/to/action_dataset \
SUBTASK_LABEL_DIR=/path/to/offline_labels \
BASE_VLM=/path/to/Qwen3-VL-4B-Instruct \
bash scripts/train_subtask_m2w_deepspeed.sh \
  starVLA/config/training/subtask_m2w_robotwin_w2.yaml
```

## Checkpoint Contract

Evaluation requires:

```text
<run-directory>/
  config.yaml
  dataset_statistics.json
  checkpoints/
    steps_<N>_pytorch_model.pt
```

`dataset_statistics.json` must come from the action dataset used for training.
It provides the normalization statistics needed to convert policy outputs back
to the dataset action space. Do not silently reuse statistics from a different
dataset or action representation.

Optimizer, scheduler, and RNG states are optional for evaluation but required
for a full training resume.

## Release Policy

Release large artifacts in a dedicated model or dataset repository. Include:

- the exact source-code revision;
- the training configuration;
- dataset and label version identifiers;
- checkpoint step and action representation;
- normalization statistics;
- benchmark and inference settings.

Never commit API keys, access tokens, private dataset paths, or credentials.
