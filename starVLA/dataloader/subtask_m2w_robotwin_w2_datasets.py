"""
Robotwin W2 M2W dataloader.

This wrapper keeps the existing Robotwin M2W sampling path but tightens the
sample contract required by the W2 JEPA branch: two wrist history clips, two
future wrist clips, and an explicit JEPA loss weight.
"""

from __future__ import annotations

from pathlib import Path

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.subtask_m2w_robotwin_datasets import (
    SubtaskM2WRobotwinMixDataset,
    _cfg_bool,
    _cfg_get,
    collate_fn,
)


class SubtaskM2WRobotwinW2MixDataset(SubtaskM2WRobotwinMixDataset):
    def __init__(self, *args, data_cfg=None, **kwargs):
        super().__init__(*args, data_cfg=data_cfg, **kwargs)
        self.num_wrist_views = int(_cfg_get(data_cfg, "num_wrist_views", 2) or 2)
        if self.num_wrist_views != 2:
            raise ValueError(
                "Robotwin W2 expects exactly two wrist views "
                f"(left/right), got num_wrist_views={self.num_wrist_views}."
            )
        if not self.load_wrist_future_views:
            raise ValueError(
                "Robotwin W2 requires datasets.vla_data.load_wrist_future_views=true "
                "so JEPA has future wrist latent targets."
            )
        if self.wrist_history_frames <= 1:
            raise ValueError(
                "Robotwin W2 requires datasets.vla_data.wrist_history_frames > 1 "
                "to build V-JEPA clips."
            )

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        self._validate_wrist_views(sample, "wrist_views")
        self._validate_wrist_views(sample, "future_wrist_views")
        sample["jepa_loss_weight"] = float(sample.get("future_wrist_loss_weight", 1.0))
        sample["num_wrist_views"] = self.num_wrist_views
        return sample

    def _validate_wrist_views(self, sample: dict, key: str) -> None:
        views = sample.get(key)
        if not isinstance(views, list) or len(views) != self.num_wrist_views:
            raise ValueError(
                f"Robotwin W2 sample field `{key}` must contain exactly "
                f"{self.num_wrist_views} wrist clips, got {type(views).__name__} "
                f"with length {len(views) if isinstance(views, list) else 'n/a'}."
            )
        for view_idx, clip in enumerate(views):
            if not isinstance(clip, list) or len(clip) != self.wrist_history_frames:
                raise ValueError(
                    f"Robotwin W2 `{key}` view {view_idx} must be a clip of "
                    f"{self.wrist_history_frames} frames, got "
                    f"{len(clip) if isinstance(clip, list) else 'n/a'}."
                )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> SubtaskM2WRobotwinW2MixDataset:
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    skip_missing_datasets = _cfg_bool(_cfg_get(data_cfg, "skip_missing_datasets", False), False)
    if skip_missing_datasets:
        available_mixture_spec = []
        missing_datasets = []
        for d_name, d_weight, robot_type in mixture_spec:
            if (data_root_dir / d_name).exists():
                available_mixture_spec.append((d_name, d_weight, robot_type))
            else:
                missing_datasets.append(d_name)
        if missing_datasets:
            print(
                "Skipping missing Robotwin W2 datasets: "
                + ", ".join(sorted(set(missing_datasets)))
            )
        if not available_mixture_spec:
            raise FileNotFoundError(
                f"No datasets from mix {data_mix!r} exist under {data_root_dir}"
            )
        mixture_spec = available_mixture_spec

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    data_root_dir,
                    d_name,
                    robot_type,
                    delete_pause_frame=delete_pause_frame,
                    data_cfg=data_cfg,
                ),
                d_weight,
            )
        )

    return SubtaskM2WRobotwinW2MixDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
