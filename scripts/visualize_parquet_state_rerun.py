#!/usr/bin/env python3
"""Export a LeRobot/Robotwin parquet observation.state sequence to Rerun.

Example:
  python scripts/visualize_parquet_state_rerun.py \
    playground/Datasets/Real-World-M2W-Demo/place_bag/data/chunk-000/episode_000000.parquet \
    --output logs/rerun/place_bag_episode_000000_state.rrd

Then open:
  rerun logs/rerun/place_bag_episode_000000_state.rrd
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_STATE_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


def stack_vector_column(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df:
        raise KeyError(f"Missing parquet column: {key}")
    values = [np.asarray(v, dtype=np.float32) for v in df[key].to_numpy()]
    if not values:
        raise ValueError(f"Column {key} is empty")
    width = values[0].shape[0]
    bad = [i for i, v in enumerate(values) if v.ndim != 1 or v.shape[0] != width]
    if bad:
        raise ValueError(f"Column {key} has inconsistent vector shapes at rows: {bad[:8]}")
    return np.stack(values, axis=0)


def names_for_dim(dim: int) -> list[str]:
    if dim == len(DEFAULT_STATE_NAMES):
        return DEFAULT_STATE_NAMES
    return [f"state_{i:02d}" for i in range(dim)]


def log_scalar_series(rr, base: str, values: np.ndarray, names: Iterable[str], frames, times) -> None:
    scalar_cls = getattr(rr, "Scalar", None) or getattr(rr, "Scalars")
    for row_idx, (frame, timestamp) in enumerate(zip(frames, times)):
        rr.set_time("frame", sequence=int(frame))
        rr.set_time("time", duration=float(timestamp))
        for dim_idx, name in enumerate(names):
            rr.log(f"{base}/{name}", scalar_cls(float(values[row_idx, dim_idx])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Episode parquet file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .rrd path. Defaults to logs/rerun/<parquet-stem>_state.rrd.",
    )
    parser.add_argument(
        "--app-id",
        default="m2w-parquet-state",
        help="Rerun app id recorded in the .rrd.",
    )
    parser.add_argument(
        "--include-action",
        action="store_true",
        help="Also log action vectors, if present, under action/...",
    )
    args = parser.parse_args()

    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing rerun-sdk. Install it in the current Python env with:\n"
            "  pip install rerun-sdk\n"
        ) from exc

    parquet = args.parquet.expanduser().resolve()
    if not parquet.exists():
        raise FileNotFoundError(parquet)

    output = args.output
    if output is None:
        output = Path("logs/rerun") / f"{parquet.stem}_state.rrd"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet)
    state = stack_vector_column(df, "observation.state")
    state_names = names_for_dim(state.shape[1])

    if "frame_index" in df:
        frames = df["frame_index"].to_numpy()
    else:
        frames = np.arange(len(df))

    if "timestamp" in df:
        times = df["timestamp"].to_numpy(dtype=np.float64)
    else:
        times = frames.astype(np.float64)

    rr.init(args.app_id, spawn=False)
    rr.save(str(output))
    rr.log("episode/source_parquet", rr.TextDocument(str(parquet)))
    rr.log(
        "episode/schema",
        rr.TextDocument(
            "observation.state layout: "
            + ", ".join(f"{i}:{name}" for i, name in enumerate(state_names))
        ),
    )
    log_scalar_series(rr, "observation/state", state, state_names, frames, times)

    if args.include_action and "action" in df:
        action = stack_vector_column(df, "action")
        action_names = names_for_dim(action.shape[1])
        log_scalar_series(rr, "action", action, action_names, frames, times)

    print(f"Wrote {output}")
    print(f"rows={len(df)} state_dim={state.shape[1]}")
    print("state_names=" + ", ".join(state_names))


if __name__ == "__main__":
    main()
