#!/usr/bin/env python3
"""Convert split RTC table_cleaning episodes into Robotwin/LeRobot-style metadata.

This script does not generate CoT labels. It creates a training-friendly copy of
an already split RTC episode directory by adding the metadata files expected by
the StarVLA Robotwin dataloader.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Split RTC task directory, e.g. Real-World-Episodes/table_cleaning",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Converted Robotwin-style task directory to create",
    )
    parser.add_argument("--task-name", default="table_cleaning")
    parser.add_argument(
        "--converted-from",
        default=None,
        help="Source dataset id/name to store in meta/info.json.",
    )
    parser.add_argument(
        "--config-key",
        default=None,
        help="Optional config key for meta/steps_data_index.pkl.",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks for data/videos by default to avoid duplicating large files.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def clone_tree(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        raise FileExistsError(dst)
    dst.mkdir(parents=True)
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        out_root = dst / rel
        out_root.mkdir(parents=True, exist_ok=True)
        for dname in dirs:
            (out_root / dname).mkdir(exist_ok=True)
        for fname in files:
            copy_or_link(root_path / fname, out_root / fname, mode)


def as_array_matrix(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(v, dtype=np.float64) for v in series.to_list()], axis=0)
    return np.asarray(series.to_numpy(dtype=np.float64)).reshape(-1, 1)


def stats_for_matrix(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(values, axis=0).astype(float).tolist(),
        "std": np.std(values, axis=0).astype(float).tolist(),
        "min": np.min(values, axis=0).astype(float).tolist(),
        "max": np.max(values, axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def collect_episode_files(task_dir: Path) -> list[Path]:
    files = sorted((task_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {task_dir / 'data'}")
    return files


def load_episode_records(meta_dir: Path, episode_files: list[Path]) -> list[dict[str, Any]]:
    records_by_index: dict[int, dict[str, Any]] = {}
    episodes_jsonl = meta_dir / "episodes.jsonl"
    if episodes_jsonl.exists():
        with episodes_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                records_by_index[int(row["episode_index"])] = row

    rows: list[dict[str, Any]] = []
    for path in episode_files:
        episode_index = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path)
        existing = records_by_index.get(episode_index, {})
        task = existing.get("task") or existing.get("tasks") or "table_cleaning"
        if isinstance(task, list):
            task_list = [str(x) for x in task]
        else:
            task_list = [str(task)]
        rows.append(
            {
                "episode_index": episode_index,
                "tasks": task_list,
                "length": int(len(df)),
            }
        )
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def build_tasks(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    tasks: list[dict[str, Any]] = []
    for row in episode_rows:
        task = row["tasks"][0]
        if task not in seen:
            seen[task] = len(seen)
            tasks.append({"task_index": seen[task], "task": task})
    return tasks


def build_modality() -> dict[str, Any]:
    return {
        "action": {
            "left_joints": {"start": 0, "end": 6, "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
            "right_joints": {"start": 7, "end": 13, "original_key": "action"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
        },
        "state": {
            "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
            "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
        },
        "video": {
            "cam_high": {"original_key": "observation.images.cam_high"},
            "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def normalize_feature_info(info: dict[str, Any], fps: int) -> dict[str, Any]:
    features = dict(info.get("features", {}))
    for key in ("observation.state", "action", "observation.velocity", "observation.effort"):
        if key in features:
            features[key]["dtype"] = features[key].get("dtype", "float32")
            if "shape" not in features[key]:
                features[key]["shape"] = [14]
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key in features and "shape" not in features[key]:
            features[key]["shape"] = [1]

    for video_key in VIDEO_KEYS:
        if video_key not in features:
            features[video_key] = {}
        shape = list(features[video_key].get("shape", [480, 640, 3]))
        if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
            shape = [shape[1], shape[2], shape[0]]
        features[video_key].update(
            {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": int(shape[0]),
                    "video.width": int(shape[1]),
                    "video.channels": int(shape[2]),
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        )
    return features


def build_info(
    src_info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    fps: int,
    task_name: str,
    converted_from: str,
) -> dict[str, Any]:
    total_episodes = len(episode_rows)
    total_frames = int(sum(row["length"] for row in episode_rows))
    return {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "aloha"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_rows),
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": 1,
        "chunks_size": int(src_info.get("chunks_size", 1000)),
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": normalize_feature_info(src_info, fps),
        "dataset_name": task_name,
        "converted_from": converted_from,
    }


def compute_stats(episode_files: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    stat_keys: list[str] = []
    all_values: dict[str, list[np.ndarray]] = {}
    episode_stats: list[dict[str, Any]] = []
    total_frames = 0

    for path in episode_files:
        episode_index = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path)
        total_frames += int(len(df))
        keys = [
            c
            for c in df.columns
            if c
            in {
                "observation.state",
                "action",
                "observation.velocity",
                "observation.effort",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            }
        ]
        stat_keys = list(dict.fromkeys([*stat_keys, *keys]))
        ep_row: dict[str, Any] = {
            "episode_index": episode_index,
            "length": int(len(df)),
            "stats": {},
        }
        for key in keys:
            values = as_array_matrix(df[key])
            all_values.setdefault(key, []).append(values)
            ep_row["stats"][key] = stats_for_matrix(values)
        episode_stats.append(ep_row)

    dataset_stats = {
        key: stats_for_matrix(np.concatenate(chunks, axis=0))
        for key, chunks in all_values.items()
    }
    episode_stats.sort(key=lambda r: r["episode_index"])
    return dataset_stats, episode_stats, total_frames


def build_steps_index(episode_rows: list[dict[str, Any]], config_key: str) -> dict[str, Any]:
    steps = [
        (int(row["episode_index"]), step)
        for row in episode_rows
        for step in range(int(row["length"]))
    ]
    return {
        "config_key": config_key,
        "steps": steps,
        "num_trajectories": len(episode_rows),
        "total_steps": len(steps),
        "computed_timestamp": datetime.now(timezone.utc).isoformat(),
        "delete_pause_frame": False,
    }


def verify_converted(task_dir: Path, episode_rows: list[dict[str, Any]], total_frames: int) -> None:
    for rel in (
        "meta/modality.json",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/stats_gr00t.json",
        "meta/steps_data_index.pkl",
    ):
        path = task_dir / rel
        if not path.exists():
            raise FileNotFoundError(path)

    parquet_count = len(collect_episode_files(task_dir))
    if parquet_count != len(episode_rows):
        raise RuntimeError(f"Expected {len(episode_rows)} parquets, found {parquet_count}")

    for video_key in VIDEO_KEYS:
        video_count = len(list((task_dir / "videos" / "chunk-000" / video_key).glob("episode_*.mp4")))
        if video_count != len(episode_rows):
            raise RuntimeError(f"{video_key}: expected {len(episode_rows)} videos, found {video_count}")

    with (task_dir / "meta" / "steps_data_index.pkl").open("rb") as f:
        steps_index = pickle.load(f)
    if int(steps_index["total_steps"]) != int(total_frames):
        raise RuntimeError(f"steps total mismatch: {steps_index['total_steps']} != {total_frames}")


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)

    clone_tree(src, dst, args.copy_mode)

    src_info = read_json(src / "meta" / "info.json")
    fps = int(args.fps or src_info.get("fps", 50))
    episode_files = collect_episode_files(dst)
    episode_rows = load_episode_records(dst / "meta", episode_files)
    task_rows = build_tasks(episode_rows)
    dataset_stats, episode_stats, total_frames = compute_stats(episode_files)
    if total_frames != sum(int(row["length"]) for row in episode_rows):
        raise RuntimeError("Computed stats frame count does not match episodes.jsonl lengths")

    write_json(dst / "meta" / "modality.json", build_modality())
    converted_from = args.converted_from or f"realworld episode dataset {args.task_name}"
    config_key = args.config_key or f"{args.task_name}_robotwin_format"
    write_json(
        dst / "meta" / "info.json",
        build_info(src_info, episode_rows, task_rows, fps, args.task_name, converted_from),
    )
    write_jsonl(dst / "meta" / "tasks.jsonl", task_rows)
    write_jsonl(dst / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(dst / "meta" / "episodes_stats.jsonl", episode_stats)
    write_json(dst / "meta" / "stats_gr00t.json", dataset_stats)
    with (dst / "meta" / "steps_data_index.pkl").open("wb") as f:
        pickle.dump(build_steps_index(episode_rows, config_key), f)

    verify_converted(dst, episode_rows, total_frames)
    summary = {
        "src": str(src),
        "dst": str(dst),
        "episodes": len(episode_rows),
        "frames": total_frames,
        "tasks": len(task_rows),
        "fps": fps,
        "copy_mode": args.copy_mode,
        "status": "ok",
    }
    write_json(dst / "conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
