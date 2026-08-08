# Script Guide

The `scripts/` directory contains the public training entry points and a small
set of runtime preparation and validation utilities.

## Training Entry Points

- `train_subtask_m2w_deepspeed.sh`: shared LIBERO and RoboTwin W²-VLA
  launcher; pass the corresponding YAML configuration as its first argument.
- `train_subtask_m2w_realworld_*.sh`: single-task real-world LeRobot
  launchers built on the RoboTwin trainer and data adapter.
- `env_robotwin.sh`: optional RoboTwin environment defaults.

All paths and account-specific settings can be overridden with environment
variables. Keep credentials such as `WANDB_API_KEY` in the shell environment;
do not write them into scripts or configuration files.

Training defaults to `WANDB_MODE=offline`. Set `WANDB_MODE=online` explicitly
after authenticating with W&B when live synchronization is desired.

`PRETRAINED_CHECKPOINT` performs weight-only initialization and starts a new
training stage at step zero. `RESUME=true` resumes the latest checkpoint in the
run directory and restores optimizer, scheduler, RNG, and global-step state by
default. These two modes are intentionally separate.

## Data Conversion

The remaining conversion helpers prepare real-world demonstrations in the
LeRobot-compatible layout expected by the training dataloader. They are not
imported by the training runtime.

## Validation

`check_vjepa2_encoder.sh` and `verify_m2w_future_alignment.py` provide focused
checks for the frozen visual encoder and temporal alignment. Dataset and
checkpoint placement is described in `docs/DATA_AND_CHECKPOINTS.md`.
