#!/usr/bin/env python3
"""Materialize RTC-Anything per-episode video clips.

`prepare_rtc_anything_episode_dataset.py` splits the low-dimensional parquet and
writes exact source video ranges into per-episode metadata.  This script reads
that metadata and writes RoboTwin-style per-episode mp4 files:

videos/chunk-000/observation.images.cam_high/episode_000000.mp4
videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4

It decodes each long source mp4 once per camera with PyAV, which is important for
RTC-Anything's AV1 videos on machines without a system ffmpeg binary.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class ClipSpec:
    task_name: str
    episode_index: int
    view_name: str
    video_key: str
    source_video: Path
    start_frame: int
    end_frame: int
    dst: Path
    fps: float
    episode_json: Path

    @property
    def expected_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


class ClipWriter:
    def __init__(self, spec: ClipSpec, codec: str, crf: int, preset: str) -> None:
        import av

        self.spec = spec
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.container = av.open(str(spec.dst), "w")
        self.stream = None
        self.written = 0

    def write(self, frame: Any) -> None:
        if self.stream is None:
            self.stream = self.container.add_stream(self.codec, rate=self.spec.fps)
            self.stream.width = frame.width
            self.stream.height = frame.height
            self.stream.pix_fmt = "yuv420p"
            if self.codec in {"h264", "libx264"}:
                self.stream.options = {"preset": self.preset, "crf": str(self.crf)}
        frame = frame.reformat(format="yuv420p")
        # Decoded frames keep timestamps from the long source video.  Clear both
        # PTS and time_base so the encoder writes a local 0-based episode
        # timeline that browser video controls and frame readers can seek.
        frame.pts = None
        frame.time_base = None
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.written += 1

    def close(self) -> int:
        if self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()
        return self.written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut RTC-Anything episode videos from prepared episode manifests."
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=Path("playground/Datasets/Real-World-Episodes"),
        help="Prepared episode dataset root.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task folders to materialize. Defaults to all tasks in manifest.",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        help="Optional subset of episode indices for debugging.",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="Output codec for per-episode mp4 clips. libx264 is browser-friendly.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality when using libx264/h264. Lower is higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="H.264 encoding preset when using libx264/h264.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clips.")
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="Fill episode_video/materialized_num_frames fields in episode JSON and manifest.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Decode written clips and verify frame counts after materialization.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_specs(args: argparse.Namespace) -> list[ClipSpec]:
    root = args.episode_root.resolve()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    wanted_tasks = set(args.tasks) if args.tasks else None
    wanted_episodes = set(args.episode_indices) if args.episode_indices else None
    specs: list[ClipSpec] = []

    for record in read_jsonl(manifest_path):
        task_name = record["task_name"]
        episode_index = int(record["episode_index"])
        if wanted_tasks is not None and task_name not in wanted_tasks:
            continue
        if wanted_episodes is not None and episode_index not in wanted_episodes:
            continue
        episode_json = root / task_name / record["episode_json"]
        fps = float(record.get("fps", 50))
        for view_name, video_record in record["videos"].items():
            video_key = video_record["video_key"]
            if video_key not in VIDEO_KEYS:
                continue
            rel = Path("videos/chunk-000") / video_key / f"episode_{episode_index:06d}.mp4"
            specs.append(
                ClipSpec(
                    task_name=task_name,
                    episode_index=episode_index,
                    view_name=view_name,
                    video_key=video_key,
                    source_video=Path(video_record["source_video_abs"]),
                    start_frame=int(video_record["source_frame_start"]),
                    end_frame=int(video_record["source_frame_end"]),
                    dst=root / task_name / rel,
                    fps=fps,
                    episode_json=episode_json,
                )
            )
    return specs


def group_by_source(specs: list[ClipSpec]) -> dict[Path, list[ClipSpec]]:
    grouped: dict[Path, list[ClipSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.source_video, []).append(spec)
    for source_specs in grouped.values():
        source_specs.sort(key=lambda s: (s.start_frame, s.end_frame, s.task_name, s.episode_index))
    return grouped


def remove_or_skip_existing(spec: ClipSpec, overwrite: bool) -> bool:
    if not spec.dst.exists():
        return False
    if overwrite:
        spec.dst.unlink()
        return False
    return True


def materialize_source(
    source: Path,
    specs: list[ClipSpec],
    codec: str,
    crf: int,
    preset: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    import av

    results: list[dict[str, Any]] = []
    pending = [spec for spec in specs if not remove_or_skip_existing(spec, overwrite)]
    skipped = [spec for spec in specs if spec.dst.exists() and not overwrite]
    for spec in skipped:
        results.append(
            {
                "task_name": spec.task_name,
                "episode_index": spec.episode_index,
                "view_name": spec.view_name,
                "video_key": spec.video_key,
                "dst": str(spec.dst),
                "expected_frames": spec.expected_frames,
                "written_frames": None,
                "status": "skipped_exists",
            }
        )
    if not pending:
        return results

    pending.sort(key=lambda s: s.start_frame)
    max_end = max(spec.end_frame for spec in pending)
    active: list[tuple[ClipSpec, ClipWriter]] = []
    next_idx = 0
    container = av.open(str(source))
    try:
        for frame_idx, frame in enumerate(container.decode(video=0)):
            while next_idx < len(pending) and pending[next_idx].start_frame == frame_idx:
                spec = pending[next_idx]
                spec.dst.parent.mkdir(parents=True, exist_ok=True)
                active.append((spec, ClipWriter(spec, codec, crf, preset)))
                next_idx += 1

            still_active: list[tuple[ClipSpec, ClipWriter]] = []
            for spec, writer in active:
                if spec.start_frame <= frame_idx < spec.end_frame:
                    writer.write(frame)
                if frame_idx + 1 >= spec.end_frame:
                    written = writer.close()
                    status = "ok" if written == spec.expected_frames else "frame_count_mismatch"
                    results.append(
                        {
                            "task_name": spec.task_name,
                            "episode_index": spec.episode_index,
                            "view_name": spec.view_name,
                            "video_key": spec.video_key,
                            "dst": str(spec.dst),
                            "expected_frames": spec.expected_frames,
                            "written_frames": written,
                            "status": status,
                        }
                    )
                else:
                    still_active.append((spec, writer))
            active = still_active

            if frame_idx + 1 >= max_end and next_idx >= len(pending) and not active:
                break
    finally:
        for _, writer in active:
            writer.close()
        container.close()

    completed = {
        (r["task_name"], r["episode_index"], r["view_name"])
        for r in results
        if r["status"] in {"ok", "frame_count_mismatch"}
    }
    for spec in pending:
        key = (spec.task_name, spec.episode_index, spec.view_name)
        if key not in completed:
            results.append(
                {
                    "task_name": spec.task_name,
                    "episode_index": spec.episode_index,
                    "view_name": spec.view_name,
                    "video_key": spec.video_key,
                    "dst": str(spec.dst),
                    "expected_frames": spec.expected_frames,
                    "written_frames": 0,
                    "status": "not_reached",
                }
            )
    return results


def count_video_frames(path: Path) -> int:
    import av

    count = 0
    container = av.open(str(path))
    try:
        for _ in container.decode(video=0):
            count += 1
    finally:
        container.close()
    return count


def update_metadata(root: Path, specs: list[ClipSpec], results: list[dict[str, Any]]) -> None:
    by_key = {
        (r["task_name"], int(r["episode_index"]), r["view_name"]): r
        for r in results
        if r["status"] in {"ok", "skipped_exists"}
    }

    json_paths = sorted({spec.episode_json for spec in specs})
    for path in json_paths:
        record = load_json(path)
        task_name = record["task_name"]
        episode_index = int(record["episode_index"])
        changed = False
        for view_name, video_record in record["videos"].items():
            key = (task_name, episode_index, view_name)
            if key not in by_key:
                continue
            result = by_key[key]
            dst = Path(result["dst"])
            video_record["episode_video"] = str(dst.relative_to(root / task_name))
            if result["written_frames"] is not None:
                video_record["materialized_num_frames"] = int(result["written_frames"])
            changed = True
        if changed:
            write_json(path, record)

    # Rebuild root and task manifests from episode JSON to keep paths coherent.
    task_records: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((root).glob("*/meta/episodes/chunk-000/episode_*.json")):
        record = load_json(path)
        task_name = record["task_name"]
        record["episode_json"] = str(path.relative_to(root / task_name))
        task_records.setdefault(task_name, []).append(record)
    root_records: list[dict[str, Any]] = []
    for task_name, records in sorted(task_records.items()):
        records.sort(key=lambda r: int(r["episode_index"]))
        write_jsonl(root / task_name / "meta/episodes.jsonl", records)
        root_records.extend(records)
    root_records.sort(key=lambda r: (r["task_name"], int(r["episode_index"])))
    write_jsonl(root / "manifest.jsonl", root_records)


def main() -> None:
    args = parse_args()
    root = args.episode_root.resolve()
    specs = collect_specs(args)
    if not specs:
        raise RuntimeError("No clips selected.")

    grouped = group_by_source(specs)
    all_results: list[dict[str, Any]] = []
    for source, source_specs in sorted(grouped.items(), key=lambda item: str(item[0])):
        if not source.exists():
            raise FileNotFoundError(f"Missing source video: {source}")
        print(f"[rtc video] {source} clips={len(source_specs)}")
        all_results.extend(
            materialize_source(source, source_specs, args.codec, args.crf, args.preset, args.overwrite)
        )

    if args.verify:
        for result in all_results:
            if result["status"] not in {"ok", "skipped_exists"}:
                continue
            count = count_video_frames(Path(result["dst"]))
            result["verified_frames"] = count
            if count != result["expected_frames"]:
                result["status"] = "verify_frame_count_mismatch"

    if args.update_metadata:
        update_metadata(root, specs, all_results)

    summary = {
        "episode_root": str(root),
        "selected_clips": len(specs),
        "source_videos": len(grouped),
        "results": all_results,
        "status_counts": {},
    }
    for result in all_results:
        status = result["status"]
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    summary_path = root / "video_materialization_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary["status_counts"], ensure_ascii=False, indent=2))
    print(f"[rtc video] summary={summary_path}")

    bad = {k: v for k, v in summary["status_counts"].items() if k not in {"ok", "skipped_exists"}}
    if bad:
        raise SystemExit(f"Video materialization had non-ok statuses: {bad}")


if __name__ == "__main__":
    main()
