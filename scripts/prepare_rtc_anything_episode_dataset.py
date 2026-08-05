#!/usr/bin/env python3
"""Split RTC-Anything long parquet/video metadata into per-episode files.

The raw RTC-Anything layout stores one long low-dimensional parquet per task and
long mp4 files per camera.  The episode parquet under meta/episodes already
contains the authoritative row ranges and per-camera timestamp ranges.  This
script uses those ranges to write small per-episode parquet files plus JSON/JSONL
manifests that point to the correct video slices.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TASKS = ("table_cleaning",)
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an RTC-Anything dataset into per-episode parquet files."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("playground/Datasets/Real-World"),
        help="Downloaded RTC-Anything root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("playground/Datasets/Real-World-Episodes"),
        help="Output root for per-episode files.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task folders to split.",
    )
    parser.add_argument(
        "--dataset-id",
        default="RTC-Anything",
        help="Source dataset id written into per-episode metadata.",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        help="Optional subset of episode indices for debugging.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        help="Optional maximum number of episodes per task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output root.",
    )
    parser.add_argument(
        "--extract-videos",
        action="store_true",
        help=(
            "Physically cut per-episode mp4 clips with ffmpeg. By default the "
            "script only writes precise video slice manifests, which avoids "
            "keyframe/re-encoding drift."
        ),
    )
    parser.add_argument(
        "--video-codec",
        choices=("h264", "copy"),
        default="h264",
        help="Codec mode used only with --extract-videos.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable used only with --extract-videos.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")


def load_info(task_dir: Path) -> dict[str, Any]:
    info_path = task_dir / "meta/info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_sidecars(raw_task_dir: Path, out_task_dir: Path) -> None:
    """Copy task-level metadata, but not the long raw episode/data parquet."""
    for rel in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet"):
        src = raw_task_dir / rel
        if src.exists():
            dst = out_task_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def read_episode_index(raw_task_dir: Path) -> pd.DataFrame:
    episode_files = sorted((raw_task_dir / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet found under {raw_task_dir / 'meta/episodes'}")
    frames = [pd.read_parquet(path) for path in episode_files]
    episodes = pd.concat(frames, ignore_index=True)
    return episodes.sort_values("episode_index").reset_index(drop=True)


def data_path(raw_task_dir: Path, chunk_index: int, file_index: int) -> Path:
    return raw_task_dir / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"


def video_path(raw_task_dir: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return (
        raw_task_dir
        / "videos"
        / video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def extract_video_slice(
    *,
    ffmpeg_bin: str,
    src: Path,
    dst: Path,
    start_s: float,
    end_s: float,
    start_frame: int,
    end_frame: int,
    fps: float,
    codec: str,
    overwrite: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    if shutil.which(ffmpeg_bin) is None:
        extract_video_slice_cv2(
            src=src,
            dst=dst,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            overwrite=overwrite,
        )
        return
    duration = max(0.0, end_s - start_s)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-an",
    ]
    if codec == "copy":
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


def extract_video_slice_cv2(
    *,
    src: Path,
    dst: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    overwrite: bool,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is not available and OpenCV cannot be imported for video extraction"
        ) from exc

    if dst.exists() and overwrite:
        dst.unlink()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open source video: {src}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"OpenCV failed to read frame {start_frame} from {src}")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV failed to create video writer: {dst}")

    frame_idx = start_frame
    try:
        while frame_idx < end_frame:
            writer.write(frame)
            frame_idx += 1
            if frame_idx >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"OpenCV stopped at frame {frame_idx}, expected {end_frame}, source={src}"
                )
    finally:
        writer.release()
        cap.release()


def select_episodes(episodes: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = episodes
    if args.episode_indices:
        wanted = set(args.episode_indices)
        selected = selected[selected["episode_index"].isin(wanted)]
    if args.limit_episodes is not None:
        selected = selected.head(args.limit_episodes)
    return selected.reset_index(drop=True)


def build_video_record(
    *,
    raw_task_dir: Path,
    out_task_dir: Path,
    task_name: str,
    episode_index: int,
    row: pd.Series,
    fps: float,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for video_key in VIDEO_KEYS:
        prefix = f"videos/{video_key}"
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        start_s = float(row[f"{prefix}/from_timestamp"])
        end_s = float(row[f"{prefix}/to_timestamp"])
        start_frame = int(round(start_s * fps))
        end_frame = int(round(end_s * fps))
        src = video_path(raw_task_dir, video_key, chunk_index, file_index)
        short_view = video_key.split(".")[-1]
        clip_rel = Path("videos/chunk-000") / video_key / f"episode_{episode_index:06d}.mp4"
        clip_path = out_task_dir / clip_rel
        if args.extract_videos:
            if not src.exists():
                raise FileNotFoundError(f"Missing source video: {src}")
            extract_video_slice(
                ffmpeg_bin=args.ffmpeg_bin,
                src=src,
                dst=clip_path,
                start_s=start_s,
                end_s=end_s,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                codec=args.video_codec,
                overwrite=args.overwrite,
            )
        videos[short_view] = {
            "video_key": video_key,
            "source_video": str(src.relative_to(raw_task_dir)),
            "source_video_abs": str(src.resolve()),
            "chunk_index": chunk_index,
            "file_index": file_index,
            "from_timestamp": start_s,
            "to_timestamp": end_s,
            "duration": end_s - start_s,
            "source_frame_start": start_frame,
            "source_frame_end": end_frame,
            "expected_num_frames": end_frame - start_frame,
            "episode_video": str(clip_rel) if args.extract_videos else None,
        }
    return videos


def split_task(task_name: str, args: argparse.Namespace) -> dict[str, Any]:
    raw_task_dir = args.raw_root / task_name
    out_task_dir = args.out_root / task_name
    if not raw_task_dir.exists():
        raise FileNotFoundError(f"Missing task directory: {raw_task_dir}")

    info = load_info(raw_task_dir)
    fps = float(info.get("fps", 50))
    episodes = select_episodes(read_episode_index(raw_task_dir), args)
    copy_sidecars(raw_task_dir, out_task_dir)

    task_manifest = out_task_dir / "meta/episodes.jsonl"
    if task_manifest.exists() and args.overwrite:
        task_manifest.unlink()

    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    total_rows = 0
    warnings: list[str] = []

    for _, row in episodes.iterrows():
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        data_chunk = int(row["data/chunk_index"])
        data_file = int(row["data/file_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        key = (data_chunk, data_file)
        if key not in data_cache:
            src_data_path = data_path(raw_task_dir, data_chunk, data_file)
            if not src_data_path.exists():
                raise FileNotFoundError(f"Missing source data parquet: {src_data_path}")
            data_cache[key] = pd.read_parquet(src_data_path)
        data_df = data_cache[key]
        ep_df = data_df.iloc[start:end].copy()
        if len(ep_df) != length:
            raise ValueError(
                f"{task_name} episode {episode_index}: row count {len(ep_df)} != length {length}"
            )

        out_data_rel = Path("data/chunk-000") / f"episode_{episode_index:06d}.parquet"
        out_data_path = out_task_dir / out_data_rel
        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(out_data_path, index=False)
        total_rows += len(ep_df)

        videos = build_video_record(
            raw_task_dir=raw_task_dir,
            out_task_dir=out_task_dir,
            task_name=task_name,
            episode_index=episode_index,
            row=row,
            fps=fps,
            args=args,
        )
        for view_name, record in videos.items():
            if record["expected_num_frames"] != length:
                warnings.append(
                    f"{task_name} episode {episode_index} {view_name}: "
                    f"video frames {record['expected_num_frames']} != length {length}"
                )

        tasks_value = row.get("tasks")
        instruction = tasks_value[0] if isinstance(tasks_value, list) and tasks_value else tasks_value
        episode_record = {
            "dataset": args.dataset_id,
            "task_name": task_name,
            "episode_index": episode_index,
            "instruction": instruction,
            "tasks": tasks_value,
            "fps": fps,
            "length": length,
            "data": {
                "episode_parquet": str(out_data_rel),
                "source_data": str(data_path(raw_task_dir, data_chunk, data_file).relative_to(raw_task_dir)),
                "source_data_abs": str(data_path(raw_task_dir, data_chunk, data_file).resolve()),
                "chunk_index": data_chunk,
                "file_index": data_file,
                "dataset_from_index": start,
                "dataset_to_index": end,
            },
            "videos": videos,
        }
        episode_json_rel = Path("meta/episodes/chunk-000") / f"episode_{episode_index:06d}.json"
        write_json(out_task_dir / episode_json_rel, episode_record)
        episode_record["episode_json"] = str(episode_json_rel)
        append_jsonl(task_manifest, episode_record)
        append_jsonl(args.out_root / "manifest.jsonl", episode_record)

    summary = {
        "task_name": task_name,
        "raw_task_dir": str(raw_task_dir),
        "out_task_dir": str(out_task_dir),
        "fps": fps,
        "episodes": int(len(episodes)),
        "rows": int(total_rows),
        "extract_videos": bool(args.extract_videos),
        "warnings": warnings,
    }
    write_json(out_task_dir / "summary.json", summary)
    return summary


def write_root_readme(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Real-World RTC-Anything Episode Split",
        "",
        "Generated by `scripts/prepare_rtc_anything_episode_dataset.py`.",
        "",
        "Each task contains small per-episode low-dimensional parquet files under:",
        "",
        "```text",
        "data/chunk-000/episode_000000.parquet",
        "```",
        "",
        "Video clips are not physically extracted unless `--extract-videos` is used.",
        "When extracted, clips follow the RoboTwin-style layout:",
        "",
        "```text",
        "videos/chunk-000/observation.images.cam_high/episode_000000.mp4",
        "videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4",
        "videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4",
        "```",
        "Per-episode JSON/JSONL manifests store the source mp4 file, file index, and",
        "per-view timestamp range from the original RTC-Anything metadata.",
        "",
        "Tasks:",
    ]
    for summary in summaries:
        lines.append(
            f"- `{summary['task_name']}`: {summary['episodes']} episodes, {summary['rows']} rows"
        )
    lines.append("")
    (args.out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.raw_root = args.raw_root.resolve()
    args.out_root = args.out_root.resolve()

    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {args.out_root} (use --overwrite)")
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    root_manifest = args.out_root / "manifest.jsonl"
    if root_manifest.exists():
        root_manifest.unlink()

    summaries = [split_task(task, args) for task in args.tasks]
    write_json(
        args.out_root / "summary.json",
        {
            "raw_root": str(args.raw_root),
            "out_root": str(args.out_root),
            "tasks": [summary["task_name"] for summary in summaries],
            "episodes": sum(int(summary["episodes"]) for summary in summaries),
            "rows": sum(int(summary["rows"]) for summary in summaries),
            "extract_videos": bool(args.extract_videos),
            "warnings": [warning for summary in summaries for warning in summary["warnings"]],
        },
    )
    write_root_readme(args, summaries)

    print(json.dumps(jsonable({"summaries": summaries}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
