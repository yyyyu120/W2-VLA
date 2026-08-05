#!/usr/bin/env python3
"""Reset per-episode RTC mp4 timestamps to start at zero.

The frame content and frame count are preserved.  This is needed when clips were
encoded from long source videos without clearing decoded frame PTS values.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="Robotwin-style RTC task directory, e.g. Real-World-Episodes-RoboTwin/table_cleaning",
    )
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def read_video_info(path: Path) -> dict[str, float | int | None]:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        first_time = None
        first_pts = None
        last_time = None
        count = 0
        for frame in container.decode(video=0):
            if first_time is None:
                first_time = frame.time
                first_pts = frame.pts
            last_time = frame.time
            count += 1
        return {
            "stream_start_time": (
                float(stream.start_time * stream.time_base)
                if stream.start_time is not None
                else None
            ),
            "first_time": first_time,
            "first_pts": first_pts,
            "last_time": last_time,
            "frames": count,
        }
    finally:
        container.close()


def reset_one(args: tuple[str, str, int, str, bool]) -> dict[str, object]:
    import av

    path_str, codec, crf, preset, verify = args
    path = Path(path_str)
    tmp = path.with_suffix(path.suffix + ".tmp_reset_pts.mp4")
    if tmp.exists():
        tmp.unlink()

    try:
        before = read_video_info(path)
        in_container = av.open(str(path))
        out_container = av.open(str(tmp), "w")
        written = 0
        try:
            in_stream = in_container.streams.video[0]
            rate = in_stream.average_rate or 50
            out_stream = None
            for frame in in_container.decode(video=0):
                if out_stream is None:
                    out_stream = out_container.add_stream(codec, rate=rate)
                    out_stream.width = frame.width
                    out_stream.height = frame.height
                    out_stream.pix_fmt = "yuv420p"
                    if codec in {"h264", "libx264"}:
                        out_stream.options = {"preset": preset, "crf": str(crf)}
                frame = frame.reformat(format="yuv420p")
                # Clear the long-video timeline carried by decoded frames.  The
                # encoder will assign a fresh local timeline starting at 0.
                frame.pts = None
                frame.time_base = None
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)
                written += 1
            if out_stream is not None:
                for packet in out_stream.encode():
                    out_container.mux(packet)
        finally:
            in_container.close()
            out_container.close()

        os.replace(tmp, path)
        after = read_video_info(path) if verify else {}
        return {
            "path": path_str,
            "before": before,
            "after": after,
            "written": written,
            "status": "ok",
        }
    except Exception as exc:  # noqa: BLE001 - return failures from workers cleanly.
        if tmp.exists():
            tmp.unlink()
        return {
            "path": path_str,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_videos(task_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for key in VIDEO_KEYS:
        videos.extend(sorted((task_dir / "videos" / "chunk-000" / key).glob("episode_*.mp4")))
    return videos


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    videos = collect_videos(task_dir)
    if not videos:
        raise FileNotFoundError(f"No episode mp4 files under {task_dir}")

    payloads = [(str(p), args.codec, args.crf, args.preset, args.verify) for p in videos]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(reset_one, payload) for payload in payloads]
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if idx % 25 == 0 or idx == len(futures):
                print(f"[reset pts] {idx}/{len(futures)}")

    bad = []
    if args.verify:
        for result in results:
            if result["status"] != "ok":
                bad.append(result["path"])
                continue
            before = result["before"]
            after = result["after"]
            if before["frames"] != after["frames"] or abs(float(after["first_time"] or 0.0)) > 1e-9:
                bad.append(result["path"])

    summary = {
        "task_dir": str(task_dir),
        "videos": len(videos),
        "status": "ok" if not bad else "bad",
        "bad": bad,
        "sample": results[:10],
    }
    out = task_dir / "video_timestamp_reset_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"videos": len(videos), "bad": len(bad), "summary": str(out)}, indent=2))
    if bad:
        raise SystemExit(f"Timestamp reset verification failed for {len(bad)} videos")


if __name__ == "__main__":
    main()
