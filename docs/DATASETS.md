# Datasets

W²-VLA keeps robot action data and offline CoT annotations in separate
directories. This avoids duplicating videos when updating text labels and makes
the exact annotation release used by a run explicit.

## Recommended Layout

```text
playground/Datasets/
├── W2-VLA-Training-Data/
│   ├── libero/<four suite directories>/
│   ├── robotwin/<50 task directories>/
│   └── real_world/
│       ├── place_bag/
│       ├── put_mango/
│       ├── table_clean/
│       └── plug_in_socket/
└── W2-VLA-CoT/
    ├── libero/
    ├── robotwin/
    └── real_world/
```

## Offline CoT Release

`W2-VLA-CoT` contains frame-aligned annotations for all three training
settings. Matching images, videos, robot states, and actions are organized in
`W2-VLA-Training-Data`.

| Split | Tasks or suites | Episodes | Frames |
| --- | ---: | ---: | ---: |
| LIBERO | 4 suites | 1,693 | 273,465 |
| RoboTwin | 50 tasks | 2,500 | 549,787 |
| Real world: place bag | 1 | 80 | 24,000 |
| Real world: put mango | 1 | 100 | 31,000 |
| Real world: table clean | 1 | 100 | 89,469 |
| Real world: plug in socket | 1 | 100 | 75,679 |
| **Total** | **58** | **4,573** | **1,043,400** |

Each episode is stored as `episode_XXXXXX.npz` with the following public
fields:

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

Task fields are included when available in the source dataset. The four CoT
arrays are frame aligned and have length `num_frames`. The default W2-VLA
training pipeline uses only `cot_train_text` for language supervision,
formatted as:

```text
Subtask: ...
Reasoning: ...
Wrist: ...
```

The decomposed CoT fields are provided for inspection, filtering, and
ablation; they are not separately consumed by the default trainer.

Download the release with:

```bash
W2_LABEL_REPO=yuuu94/W2-VLA-CoT

hf download "$W2_LABEL_REPO" \
  --repo-type dataset \
  --local-dir playground/Datasets/W2-VLA-CoT
```

Use the corresponding subdirectory as `SUBTASK_LABEL_DIR`:

```text
LIBERO:     playground/Datasets/W2-VLA-CoT/libero
RoboTwin:   playground/Datasets/W2-VLA-CoT/robotwin
Real world: playground/Datasets/W2-VLA-CoT/real_world
```

## Action Data Release

`W2-VLA-Training-Data` contains all LeRobot action datasets used by the three
training settings. Action data and CoT labels are released separately so that
text annotations can be updated without duplicating videos.

| Split | Datasets | Episodes | Frames |
| --- | ---: | ---: | ---: |
| LIBERO | 4 | 1,693 | 273,465 |
| RoboTwin | 50 | 2,500 | 549,787 |
| Real world | 4 | 380 | 220,148 |
| **Total** | **58** | **4,573** | **1,043,400** |

Download the directly expanded datasets:

```bash
W2_DATA_REPO=yuuu94/W2-VLA-Training-Data

hf download "$W2_DATA_REPO" \
  --repo-type dataset \
  --local-dir playground/Datasets/W2-VLA-Training-Data
```

Every downloaded leaf already contains the standard `meta/`, `data/`, and
`videos/` directories, so no extraction step is required. The release excludes
generation caches, review artifacts, backups, temporary files, and logs.

### LIBERO

Use `playground/Datasets/W2-VLA-Training-Data/libero` as `DATA_ROOT_DIR`.
Alternatively, the four suites can be converted independently with
[`examples/LIBERO/data_preparation.sh`](../examples/LIBERO/data_preparation.sh).

### RoboTwin

Use the 50-task clean RoboTwin LeRobot dataset under
`playground/Datasets/W2-VLA-Training-Data/robotwin`.

### Real World

Store the four LeRobot tasks under one canonical root:

```text
playground/Datasets/W2-VLA-Training-Data/real_world/
├── place_bag/
├── put_mango/
├── table_clean/
└── plug_in_socket/
```

The task wrappers accept this root through `DATA_ROOT_DIR`. Keeping all tasks
under one root removes the historical task-specific parent-directory names.

## Pairing Checks

Before training, verify that every action-data episode has exactly one CoT
file and that `num_frames` agrees with the LeRobot episode length. Do not place
backup files, generation caches, contact sheets, or review reports inside the
released label directory.
