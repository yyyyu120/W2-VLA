---
pretty_name: "World-to-Wrist: Offline CoT Labels"
language:
- en
tags:
- robotics
- vision-language-action
- chain-of-thought
- libero
- robotwin
---
# World-to-Wrist: Offline CoT Labels

This dataset contains frame-aligned offline chain-of-thought annotations used
to train W²-VLA policies on LIBERO, RoboTwin, and four real-world manipulation
tasks. Matching LeRobot action data is available in `W2-VLA-Training-Data`.

## Dataset Structure

```text
W2-VLA-CoT/
├── libero/
│   ├── libero_10_no_noops_1.0.0_lerobot/
│   ├── libero_goal_no_noops_1.0.0_lerobot/
│   ├── libero_object_no_noops_1.0.0_lerobot/
│   └── libero_spatial_no_noops_1.0.0_lerobot/
├── robotwin/
│   └── <50 task directories>/
├── real_world/
│   ├── place_bag/
│   ├── put_mango/
│   ├── table_clean/
│   └── plug_in_socket/
└── dataset_manifest.json
```

| Split | Episodes | Frames |
| --- | ---: | ---: |
| LIBERO | 1,693 | 273,465 |
| RoboTwin | 2,500 | 549,787 |
| Real world: place bag | 80 | 24,000 |
| Real world: put mango | 100 | 31,000 |
| Real world: table clean | 100 | 89,469 |
| Real world: plug in socket | 100 | 75,679 |
| **Total** | **4,573** | **1,043,400** |

## Annotation Format

Each `episode_XXXXXX.npz` file contains frame-aligned arrays. The policy is
trained with `cot_train_text`, which has the following three-field format:

```text
Subtask: ...
Reasoning: ...
Wrist: ...
```

The public schema contains:

```text
schema_version
episode_index
num_frames
task
task_description
task_name
cot_train_text
cot_subtask
cot_reasoning
cot_wrist_focus
```

Task fields are present when available. The four CoT arrays have length
`num_frames`.

## Download

```bash
hf download yuuu94/W2-VLA-CoT \
  --repo-type dataset \
  --local-dir playground/Datasets/W2-VLA-CoT
```

Use `libero/`, `robotwin/`, or `real_world/` as the label root for the matching
training configuration.
