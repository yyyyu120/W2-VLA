#!/usr/bin/env python
"""
Smoke test V-JEPA2.1 as a frozen dense encoder.

This script intentionally tests only the encoder use case required by W^2-VLA:
PIL images -> frozen V-JEPA2.1 -> dense latent tokens [B, N, D].
It does not instantiate the action model and does not train anything.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


VJEPA21_CHECKPOINT_URLS = {
    "base": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
    "large": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
    "giant": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
    "gigantic": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_test_image(size: int) -> Image.Image:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(255, 0, size, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _load_image(path: str | None, size: int) -> Image.Image:
    if not path:
        return _make_test_image(size)
    return Image.open(path).convert("RGB")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify V-JEPA2.1 frozen encoder output shape.")
    parser.add_argument("--backend", choices=["torchhub", "hf"], default="torchhub")
    parser.add_argument("--hub-repo", default="facebookresearch/vjepa2")
    parser.add_argument("--hub-model", default="vjepa2_1_vit_large_384")
    parser.add_argument("--hub-source", default="github")
    parser.add_argument("--base-encoder", default=None, help="HF-style local path or repo when --backend hf.")
    parser.add_argument("--image", default=None, help="Optional RGB image path. If omitted, a synthetic image is used.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--save-npz", default=None, help="Optional path to save output tokens for inspection.")
    parser.add_argument("--print-checkpoint-urls", action="store_true")
    parser.add_argument("--urls-only", action="store_true", help="Print checkpoint URLs and exit without importing torch.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.print_checkpoint_urls:
        print("Official V-JEPA2.1 checkpoint URLs:")
        for name, url in VJEPA21_CHECKPOINT_URLS.items():
            print(f"  {name:8s} {url}")
        print()
        if args.urls_only:
            return

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in this environment. Run this script in the starVLA/M2W "
            "training environment with torch, torchvision, timm, einops, and transformers installed."
        ) from exc

    from starVLA.model.modules.frozen_visual_encoder import FrozenVJEPA2Encoder

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Loading V-JEPA2.1 encoder...")
    print(f"  backend={args.backend}")
    print(f"  hub_repo={args.hub_repo}")
    print(f"  hub_model={args.hub_model}")
    print(f"  device={device}")

    start = time.time()
    try:
        encoder = FrozenVJEPA2Encoder(
            base_encoder=args.base_encoder,
            backend=args.backend,
            hub_repo=args.hub_repo,
            hub_model=args.hub_model,
            hub_source=args.hub_source,
            pretrained=True,
            num_frames=args.num_frames,
            image_size=args.image_size,
        ).to(device)
    except Exception as exc:
        tb = traceback.format_exc()
        if "localhost:8300" in tb:
            raise SystemExit(
                "V-JEPA2.1 torchhub code tried to download weights from http://localhost:8300, "
                "but no local file server is running there.\n\n"
                "This is a checkpoint download URL problem, not an encoder-shape problem.\n"
                "Fix for ViT-L:\n"
                "  bash scripts/download_vjepa2_1_checkpoint.sh large torchhub-cache\n"
                "Then rerun:\n"
                "  bash scripts/check_vjepa2_encoder.sh\n\n"
                "Official ViT-L URL:\n"
                f"  {VJEPA21_CHECKPOINT_URLS['large']}\n\n"
                f"Original error: {exc}"
            ) from exc
        raise
    encoder.eval()
    load_time = time.time() - start

    requires_grad = [p.requires_grad for p in encoder.parameters()]
    print(f"Loaded in {load_time:.2f}s")
    print(f"  hidden_size={encoder.hidden_size}")
    print(f"  trainable_params={sum(1 for x in requires_grad if x)} / {len(requires_grad)}")

    image = _load_image(args.image, args.image_size)
    batch = [[image.copy()] for _ in range(args.batch_size)]

    start = time.time()
    with torch.no_grad():
        tokens = encoder(batch)
    forward_time = time.time() - start

    print("Forward OK")
    print(f"  tokens.shape={tuple(tokens.shape)}")
    print(f"  tokens.dtype={tokens.dtype}")
    print(f"  tokens.device={tokens.device}")
    print(f"  tokens.requires_grad={tokens.requires_grad}")
    print(f"  finite={bool(torch.isfinite(tokens).all().item())}")
    print(f"  forward_time={forward_time:.2f}s")

    if args.save_npz:
        out_path = Path(args.save_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, tokens=tokens.detach().cpu().float().numpy())
        print(f"Saved tokens to {out_path}")


if __name__ == "__main__":
    main()
