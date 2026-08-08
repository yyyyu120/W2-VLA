#!/usr/bin/env python3
"""
Verify whether M2W predicted wrist latents align with future wrist trends.

This diagnostic runs on offline training samples and compares:
  - pred_future vs true future target
  - pred_future vs current wrist target
  - pred_future vs shuffled future target
  - (pred_future - current) vs (future - current) trend direction

The goal is to check whether the future latent branch learns a task-relevant
future direction rather than merely copying the current wrist representation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.subtask_m2w_datasets import get_vla_dataset  # noqa: E402
from starVLA.model.framework import build_framework  # noqa: E402
from starVLA.model.framework.share_tools import read_mode_config  # noqa: E402


def _set_if_not_none(cfg, dotted_key: str, value: Any) -> None:
    if value is None:
        return
    target = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def load_model(checkpoint: Path, args) -> torch.nn.Module:
    model_config, _ = read_mode_config(str(checkpoint))
    cfg = OmegaConf.create(model_config)

    _set_if_not_none(cfg, "framework.vjepa2.backend", args.vjepa_backend)
    _set_if_not_none(cfg, "framework.vjepa2.base_encoder", args.vjepa_base_encoder)
    _set_if_not_none(cfg, "framework.vjepa2.num_frames", args.vjepa_num_frames)
    _set_if_not_none(cfg, "framework.vjepa2.image_size", args.vjepa_image_size)
    _set_if_not_none(cfg, "framework.vjepa2.max_tokens", args.vjepa_max_tokens)
    _set_if_not_none(cfg, "datasets.vla_data.data_mix", args.data_mix)
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.datasets.vla_data.num_workers = 0
    cfg.trainer.pretrained_checkpoint = None

    model = build_framework(cfg)
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model_state = model.state_dict()
    missing = set(model_state) - set(state)
    action_mode = str(cfg.framework.main_to_wrist.get("action_condition_mode", "concat")).lower()
    if cfg.framework.name == "QwenSubtaskM2W" and action_mode != "residual":
        for key in list(missing):
            if key.startswith("m2w_adapter.action_residual_"):
                state[key] = model_state[key]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_dataset(model: torch.nn.Module, args):
    cfg = model.config
    _set_if_not_none(cfg, "datasets.vla_data.data_mix", args.data_mix)
    cfg.datasets.vla_data.num_workers = 0
    if args.wrist_history_frames is not None:
        cfg.datasets.vla_data.wrist_history_frames = args.wrist_history_frames
    if args.future_wrist_target is not None:
        cfg.datasets.vla_data.future_wrist_target = args.future_wrist_target
    return get_vla_dataset(data_cfg=cfg.datasets.vla_data)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(x.shape[0], -1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(flatten(a), flatten(b), dim=-1)


def smooth_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(a.float(), b.float(), reduction="none").mean(dim=tuple(range(1, a.dim())))


@torch.inference_mode()
def batch_latents(model, examples):
    (
        batch_images,
        wrist_views,
        future_wrist_views,
        instructions,
        cot_targets,
        state,
        wrist_loss_weights,
        future_loss_weights,
    ) = model.align_model_input(examples)

    hidden, h_reason, cot_loss, _ = model._qwen_forward(  # noqa: SLF001
        batch_images,
        instructions,
        cot_targets,
        compute_cot_loss=True,
    )
    adapter_dtype = next(model.m2w_adapter.parameters()).dtype
    hidden_for_condition = hidden.to(adapter_dtype)
    h_reason = h_reason.to(adapter_dtype)
    state_tensor = model._state_tensor(state, hidden.device, adapter_dtype)  # noqa: SLF001

    wrist_tokens = model.visual_encoder(wrist_views).to(hidden.device, dtype=adapter_dtype)
    adapter_out = model.m2w_adapter(
        wrist_tokens=wrist_tokens,
        reasoning_state=h_reason,
        qwen_hidden=hidden_for_condition,
        state=state_tensor,
    )
    pred = model.m2w_adapter.predict_future(
        c_wrist=adapter_out["c_wrist"],
        q_wrist=adapter_out["q_wrist"],
    )
    future_tokens = model.visual_encoder(future_wrist_views).to(hidden.device, dtype=adapter_dtype)
    future = model.m2w_adapter.extract_future_target(
        future_wrist_tokens=future_tokens,
        q_wrist=adapter_out["q_wrist"].detach(),
    )
    current = model.m2w_adapter.extract_future_target(
        future_wrist_tokens=wrist_tokens,
        q_wrist=adapter_out["q_wrist"].detach(),
    )
    return {
        "pred": pred,
        "future": future,
        "current": current,
        "cot_loss": cot_loss.detach().float(),
        "wrist_loss_weights": wrist_loss_weights,
    }


def add_records(records, examples, latents):
    pred = latents["pred"]
    future = latents["future"]
    current = latents["current"]
    batch_size = pred.shape[0]
    shuffled = future.roll(shifts=1, dims=0) if batch_size > 1 else future

    pred_future_l1 = smooth_l1(pred, future)
    pred_current_l1 = smooth_l1(pred, current)
    pred_shuffle_l1 = smooth_l1(pred, shuffled)

    pred_future_cos = cosine(pred, future)
    pred_current_cos = cosine(pred, current)
    pred_shuffle_cos = cosine(pred, shuffled)
    trend_cos = cosine(pred - current, future - current)

    pred_flat = F.normalize(flatten(pred), dim=-1)
    future_flat = F.normalize(flatten(future), dim=-1)
    sim = pred_flat @ future_flat.T
    top1 = sim.argmax(dim=1)
    diag = sim.diag()
    off_diag = sim.masked_fill(torch.eye(batch_size, dtype=torch.bool, device=sim.device), -1e9).max(dim=1).values

    for i, example in enumerate(examples):
        records.append(
            {
                "phase": str(example.get("cot_phase", "")),
                "semantic": str(example.get("semantic_motion_stage", "")),
                "contact": str(example.get("cot_contact_state", "")),
                "segment_progress": float(example.get("segment_progress", 0.0) or 0.0),
                "future_index": example.get("future_wrist_index", None),
                "subgoal_index": example.get("subgoal_wrist_index", None),
                "pred_future_l1": float(pred_future_l1[i].item()),
                "pred_current_l1": float(pred_current_l1[i].item()),
                "pred_shuffle_l1": float(pred_shuffle_l1[i].item()),
                "pred_future_cos": float(pred_future_cos[i].item()),
                "pred_current_cos": float(pred_current_cos[i].item()),
                "pred_shuffle_cos": float(pred_shuffle_cos[i].item()),
                "trend_cos": float(trend_cos[i].item()),
                "retrieval_top1": int(top1[i].item() == i),
                "retrieval_margin": float((diag[i] - off_diag[i]).item()) if batch_size > 1 else 0.0,
            }
        )


def summarize(records):
    keys = [
        "pred_future_l1",
        "pred_current_l1",
        "pred_shuffle_l1",
        "pred_future_cos",
        "pred_current_cos",
        "pred_shuffle_cos",
        "trend_cos",
        "retrieval_top1",
        "retrieval_margin",
    ]

    def stats(rows):
        out = {"n": len(rows)}
        for key in keys:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[key] = float(values.mean()) if len(values) else float("nan")
        out["future_l1_gain_vs_current"] = out["pred_current_l1"] - out["pred_future_l1"]
        out["future_cos_gain_vs_current"] = out["pred_future_cos"] - out["pred_current_cos"]
        out["future_cos_gain_vs_shuffle"] = out["pred_future_cos"] - out["pred_shuffle_cos"]
        return out

    by_phase = defaultdict(list)
    for row in records:
        by_phase[row["phase"] or "unknown"].append(row)

    return {
        "overall": stats(records),
        "by_phase": {phase: stats(rows) for phase, rows in sorted(by_phase.items())},
    }


def print_summary(summary):
    overall = summary["overall"]
    print("\n=== M2W future latent alignment ===")
    print(f"n={overall['n']}")
    print(
        "L1 lower is better: "
        f"pred->future={overall['pred_future_l1']:.4f} | "
        f"pred->current={overall['pred_current_l1']:.4f} | "
        f"pred->shuffled={overall['pred_shuffle_l1']:.4f} | "
        f"gain_vs_current={overall['future_l1_gain_vs_current']:.4f}"
    )
    print(
        "Cos higher is better: "
        f"pred~future={overall['pred_future_cos']:.4f} | "
        f"pred~current={overall['pred_current_cos']:.4f} | "
        f"pred~shuffled={overall['pred_shuffle_cos']:.4f} | "
        f"gain_vs_current={overall['future_cos_gain_vs_current']:.4f}"
    )
    print(
        "Trend direction: "
        f"cos(pred-current, future-current)={overall['trend_cos']:.4f}"
    )
    print(
        "Batch retrieval: "
        f"top1={overall['retrieval_top1']:.3f} | "
        f"margin={overall['retrieval_margin']:.4f}"
    )

    print("\nBy phase:")
    for phase, stats in summary["by_phase"].items():
        print(
            f"  {phase:>9s} n={stats['n']:4d} | "
            f"L1 future/current={stats['pred_future_l1']:.4f}/{stats['pred_current_l1']:.4f} | "
            f"cos future/current={stats['pred_future_cos']:.4f}/{stats['pred_current_cos']:.4f} | "
            f"trend={stats['trend_cos']:.4f} | top1={stats['retrieval_top1']:.3f}"
        )

    print("\nInterpretation:")
    print("  Good sign: pred->future L1 < pred->current/shuffled L1.")
    print("  Good sign: pred~future cosine > pred~current/shuffled cosine.")
    print("  Strong trend sign: trend cosine is clearly positive.")
    print("  If trend cosine is near 0 but pred~future is high, the branch may encode future semantics but not motion direction.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-mix", default=None)
    parser.add_argument("--vjepa-backend", default=None)
    parser.add_argument("--vjepa-base-encoder", default=None)
    parser.add_argument("--vjepa-num-frames", type=int, default=None)
    parser.add_argument("--vjepa-image-size", type=int, default=None)
    parser.add_argument("--vjepa-max-tokens", type=int, default=None)
    parser.add_argument("--wrist-history-frames", type=int, default=None)
    parser.add_argument("--future-wrist-target", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)
    model.eval()

    dataset = build_dataset(model, args)
    indices = np.random.default_rng(args.seed).choice(len(dataset), size=args.num_samples, replace=False)

    records = []
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        examples = [dataset[int(idx)] for idx in batch_indices]
        latents = batch_latents(model, examples)
        add_records(records, examples, latents)
        print(f"processed {len(records)}/{len(indices)}", flush=True)

    summary = summarize(records)
    print_summary(summary)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
