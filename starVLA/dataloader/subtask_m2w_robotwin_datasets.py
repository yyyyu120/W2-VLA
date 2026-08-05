"""
Robotwin M2W dataloader.

This keeps the StarVLA/GR00T LeRobot sampling and action normalization path,
but returns the extra fields needed by the M2W pipeline: current tri-view Qwen
images, left/right wrist history clips, optional future wrist clips, and
offline CoT targets.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def collate_fn(batch):
    return batch


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _cfg_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_str(value, default=""):
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    return str(value).strip()


class SubtaskM2WRobotwinMixDataset(LeRobotMixtureDataset):
    # Robotwin wrist JEPA uses an action-horizon-16 temporal window while
    # V-JEPA consumes 8 RGB frames.
    DEFAULT_WRIST_TEMPORAL_STRIDE = 2

    def __init__(self, *args, data_cfg=None, **kwargs):
        super().__init__(*args, data_cfg=data_cfg, **kwargs)
        self.data_cfg = data_cfg
        self.subtask_label_dir = Path(str(_cfg_get(data_cfg, "subtask_label_dir", ""))) if _cfg_get(data_cfg, "subtask_label_dir", "") else None
        self.wrist_history_frames = max(1, int(_cfg_get(data_cfg, "wrist_history_frames", 1) or 1))
        self.future_wrist_frames = self.wrist_history_frames
        self.wrist_temporal_stride = (
            self.DEFAULT_WRIST_TEMPORAL_STRIDE if self.wrist_history_frames > 1 else 1
        )
        self.load_wrist_future_views = _cfg_bool(
            _cfg_get(data_cfg, "load_wrist_future_views", False),
            False,
        )
        self.future_wrist_k = int(_cfg_get(data_cfg, "future_wrist_k", 16) or 16)
        self.future_small_gap = int(_cfg_get(data_cfg, "future_wrist_small_gap", 2) or 2)
        self.prefer_label_instruction = _cfg_bool(
            _cfg_get(data_cfg, "prefer_label_instruction", False),
            False,
        )
        self._label_cache = {}
        self._missing_label_paths = set()
        print(
            "SubtaskM2WRobotwinMixDataset | "
            f"subtask_label_dir={self.subtask_label_dir} | "
            f"wrist_history_frames={self.wrist_history_frames} | "
            f"future_wrist_frames={self.future_wrist_frames} | "
            f"wrist_temporal_stride={self.wrist_temporal_stride} | "
            f"future_wrist_k={self.future_wrist_k} | "
            f"load_wrist_future_views={self.load_wrist_future_views} | "
            f"prefer_label_instruction={self.prefer_label_instruction}"
        )

    @staticmethod
    def _episode_id(trajectory_id) -> int:
        try:
            return int(trajectory_id)
        except (TypeError, ValueError):
            return int(str(trajectory_id).split("_")[-1])

    def _load_label_data(self, dataset, trajectory_id):
        if self.subtask_label_dir is None:
            return None
        episode_id = self._episode_id(trajectory_id)
        label_path = self.subtask_label_dir / dataset.dataset_name / f"episode_{episode_id:06d}.npz"
        if not label_path.exists():
            if label_path not in self._missing_label_paths:
                print(f"Warning: missing Robotwin M2W label file: {label_path}")
                self._missing_label_paths.add(label_path)
            return None
        if label_path not in self._label_cache:
            try:
                with np.load(label_path, allow_pickle=True) as labels:
                    self._label_cache[label_path] = {key: labels[key] for key in labels.files}
            except Exception as exc:
                print(f"Warning: failed to load Robotwin M2W labels from {label_path}: {exc}")
                self._label_cache[label_path] = None
        return self._label_cache[label_path]

    @staticmethod
    def _label_value(label_data, key: str, step: int, default=None):
        if not label_data or key not in label_data:
            return default
        values = label_data[key]
        if np.ndim(values) == 0:
            return values.item()
        if len(values) == 0:
            return default
        index = max(0, min(int(step), len(values) - 1))
        return values[index]

    def _cot_target(self, label_data, step: int) -> str:
        for key in ("cot_train_text", "cot_text"):
            text = _as_str(self._label_value(label_data, key, step, ""))
            if text:
                return text
        subtask = _as_str(self._label_value(label_data, "cot_subtask", step, ""))
        reasoning = _as_str(self._label_value(label_data, "cot_reasoning", step, ""))
        wrist = _as_str(self._label_value(label_data, "cot_wrist_focus", step, ""))
        if subtask or reasoning or wrist:
            return f"Subtask: {subtask}. Reasoning: {reasoning}. Wrist: {wrist}."
        return ""

    def _instruction_from_label(self, label_data) -> str:
        for key in ("task_description", "task"):
            text = _as_str(self._label_value(label_data, key, 0, ""))
            if text:
                return text
        return ""

    @staticmethod
    def _clip_frame_indices(frame_indices, max_step: int) -> np.ndarray:
        clipped = np.asarray(frame_indices, dtype=np.int64)
        clipped = np.maximum(clipped, 0)
        clipped = np.minimum(clipped, int(max_step))
        return clipped

    @classmethod
    def _unique_clipped_frame_count(cls, frame_indices, max_step: int) -> int:
        clipped = cls._clip_frame_indices(frame_indices, max_step)
        return int(np.unique(clipped).size)

    @classmethod
    def _future_independent_frame_count(
        cls,
        frame_indices,
        step: int,
        max_step: int,
    ) -> int:
        clipped = cls._clip_frame_indices(frame_indices, max_step)
        future_only = clipped[clipped > int(step)]
        return int(np.unique(future_only).size)

    def _future_step_and_weight(self, _label_data, step: int, max_step: int):
        future = min(int(step) + self.future_wrist_k, max_step)
        future = max(0, min(int(future), max_step))

        gap = int(future) - int(step)
        history_indices = self._frame_indices(
            step,
            self.wrist_history_frames,
            self.wrist_temporal_stride,
        )
        future_indices = self._frame_indices(
            future,
            self.future_wrist_frames,
            self.wrist_temporal_stride,
        )
        history_count = self._unique_clipped_frame_count(history_indices, max_step)
        future_count = self._future_independent_frame_count(
            future_indices,
            step,
            max_step,
        )
        denominator = max(1, int(self.wrist_history_frames))
        weight = min(history_count, future_count) / float(denominator)
        weight = max(0.0, min(1.0, weight))
        return int(future), gap, weight

    @staticmethod
    def _frame_indices(end_step: int, frame_count: int, stride: int = 1):
        frame_count = max(1, int(frame_count))
        stride = max(1, int(stride))
        start = int(end_step) - (frame_count - 1) * stride
        return list(range(start, int(end_step) + 1, stride))

    def _load_video_frames(self, dataset, trajectory_id, video_key: str, frame_indices):
        dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
        trajectory_index = dataset.get_trajectory_index(trajectory_id)
        max_length = int(dataset.trajectory_lengths[trajectory_index])
        clipped = np.asarray(frame_indices, dtype=np.int64)
        clipped = np.maximum(clipped, 0)
        clipped = np.minimum(clipped, max_length - 1)

        key = video_key.replace("video.", "")
        video_path = dataset.get_video_path(trajectory_id, key)
        timestamps = dataset.curr_traj_data["timestamp"].to_numpy()[clipped]
        frames = get_frames_by_timestamps(
            video_path.as_posix(),
            timestamps,
            video_backend=dataset.video_backend,
            video_backend_kwargs=dataset.video_backend_kwargs,
        )
        return [Image.fromarray(frame).resize((224, 224)) for frame in frames]

    def _load_wrist_history_views(
        self,
        dataset,
        trajectory_id,
        end_step: int,
        frame_count: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        frame_count = self.wrist_history_frames if frame_count is None else frame_count
        stride = self.wrist_temporal_stride if stride is None else stride
        frame_indices = self._frame_indices(end_step, frame_count, stride)
        wrist_views = []
        for video_key in dataset.modality_keys["video"]:
            if "wrist" in video_key:
                wrist_views.append(
                    self._load_video_frames(dataset, trajectory_id, video_key, frame_indices)
                )
        return wrist_views

    def __getitem__(self, index: int) -> dict:
        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                while True:
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)

                prim_images = []
                current_wrist_images = []
                for video_key in dataset.modality_keys["video"]:
                    image = Image.fromarray(data[video_key][0]).resize((224, 224))
                    if "wrist" in video_key:
                        current_wrist_images.append(image)
                    else:
                        prim_images.append(image)
                all_images = prim_images + current_wrist_images

                language = data[dataset.modality_keys["language"][0]][0]
                action = []
                for action_key in dataset.modality_keys["action"]:
                    action.append(data[action_key])
                action = np.concatenate(action, axis=1).astype(np.float16)

                label_data = self._load_label_data(dataset, trajectory_id)
                label_instruction = self._instruction_from_label(label_data) if self.prefer_label_instruction else ""
                if label_instruction:
                    language = label_instruction
                cot_target = self._cot_target(label_data, step)
                wrist_views = self._load_wrist_history_views(
                    dataset,
                    trajectory_id,
                    step,
                    frame_count=self.wrist_history_frames,
                    stride=self.wrist_temporal_stride,
                )

                trajectory_index = dataset.get_trajectory_index(trajectory_id)
                max_step = int(dataset.trajectory_lengths[trajectory_index]) - 1
                future_step, future_gap, future_weight = self._future_step_and_weight(
                    label_data,
                    step,
                    max_step,
                )
                if self.load_wrist_future_views:
                    future_wrist_views = self._load_wrist_history_views(
                        dataset,
                        trajectory_id,
                        future_step,
                        frame_count=self.future_wrist_frames,
                        stride=self.wrist_temporal_stride,
                    )
                else:
                    future_wrist_views = wrist_views
                    future_weight = 0.0

                sample = {
                    "action": action,
                    "image": all_images,
                    "wrist_views": wrist_views,
                    "future_wrist_views": future_wrist_views,
                    "future_wrist_resolved_index": int(future_step),
                    "future_wrist_gap": int(future_gap),
                    "future_wrist_loss_weight": float(future_weight),
                    "cot_target": cot_target,
                    "cot_train_text": cot_target,
                    "lang": language,
                }

                if self.data_cfg is not None and _cfg_get(self.data_cfg, "include_state", False) not in ["False", False]:
                    state = []
                    for state_key in dataset.modality_keys["state"]:
                        state.append(data[state_key])
                    sample["state"] = np.concatenate(state, axis=1).astype(np.float16)

                return sample

            except Exception as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {exc}")
                    print("Retrying with new sample...")
                    index = random.randint(0, len(self) - 1)
                else:
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    raise last_exception


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> SubtaskM2WRobotwinMixDataset:
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
                "Skipping missing Robotwin datasets: "
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

    return SubtaskM2WRobotwinMixDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
