"""
Frozen V-JEPA2 / V-JEPA2.1 visual encoder wrapper.

The encoder is used only as a latent feature extractor. It is always frozen and
all forward passes run under torch.no_grad().
"""

from __future__ import annotations

import http.client
import urllib.error
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoVideoProcessor


class FrozenVJEPA2Encoder(nn.Module):
    def __init__(
        self,
        base_encoder: Optional[str] = None,
        backend: str = "auto",
        hub_repo: str = "facebookresearch/vjepa2:main",
        hub_model: str = "vjepa2_1_vit_large_384",
        hub_source: str = "github",
        pretrained: bool = True,
        num_frames: int = 16,
        image_size: Optional[int] = 384,
        max_tokens: Optional[int] = 256,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.backend = backend
        self.hub_repo = hub_repo
        self.hub_model = hub_model
        self.hub_source = hub_source
        self.num_frames = int(num_frames)
        self.image_size = int(image_size) if image_size else None
        self.max_tokens = int(max_tokens or 0)
        self.tubelet_size = 2

        self.processor = None
        self.encoder = None
        self._load_encoder(pretrained=pretrained)
        config = getattr(self.encoder, "config", None)
        if config is not None:
            self.tubelet_size = int(getattr(config, "tubelet_size", self.tubelet_size) or self.tubelet_size)

        hub_image_size = self.image_size or 384
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((hub_image_size, hub_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._freeze_encoder()

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.base_encoder:
            if self.base_encoder.startswith("vjepa2_1_"):
                return "torchhub"
            return "hf"
        return "torchhub"

    def _load_encoder(self, pretrained: bool) -> None:
        resolved = self._resolve_backend()
        self.backend = resolved

        if resolved == "torchhub":
            try:
                loaded = self._load_torchhub_encoder(pretrained=pretrained)
            except (http.client.RemoteDisconnected, urllib.error.URLError) as exc:
                raise RuntimeError(
                    "Failed to load V-JEPA2 via torch.hub because GitHub or the "
                    "checkpoint host was not reachable. Run "
                    "`bash scripts/check_vjepa2_encoder.sh` once to populate the "
                    "torchhub cache, or pass a local V-JEPA2 repo with "
                    "`framework.vjepa2.hub_repo=/path/to/vjepa2 "
                    "framework.vjepa2.hub_source=local`."
                ) from exc
            self.encoder = loaded[0] if isinstance(loaded, tuple) else loaded
            self.hidden_size = int(
                getattr(
                    self.encoder,
                    "embed_dim",
                    getattr(self.encoder, "num_features", 1024),
                )
            )
            return

        if resolved == "hf":
            if not self.base_encoder:
                raise ValueError("base_encoder must be set when backend='hf'.")
            self.encoder = AutoModel.from_pretrained(self.base_encoder)
            self.processor = AutoVideoProcessor.from_pretrained(self.base_encoder)
            self.hidden_size = int(getattr(self.encoder.config, "hidden_size", 1024))
            return

        raise ValueError(f"Unsupported V-JEPA backend: {resolved}")

    def _load_torchhub_encoder(self, pretrained: bool):
        try:
            return torch.hub.load(
                self.hub_repo,
                self.hub_model,
                source=self.hub_source,
                pretrained=pretrained,
            )
        except TypeError:
            return torch.hub.load(
                self.hub_repo,
                self.hub_model,
                source=self.hub_source,
            )

    def _freeze_encoder(self) -> None:
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        # V-JEPA is deliberately a frozen teacher/feature extractor.
        super().train(False)
        self._freeze_encoder()
        return self

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    def _ensure_encoder_compute_dtype(self):
        # The torchhub V-JEPA2.1 implementation mixes fp32 attention logits with
        # value tensors. Keep this frozen teacher in fp32 under DeepSpeed bf16.
        if self.backend == "torchhub" and self.dtype != torch.float32:
            self.encoder.float()
            self._freeze_encoder()

    def _frame_indices(self, num_available: int) -> np.ndarray:
        if num_available <= 0:
            raise ValueError("V-JEPA clip must contain at least one frame.")
        if num_available >= self.num_frames:
            return np.round(np.linspace(0, num_available - 1, self.num_frames)).astype(np.int64)
        return np.floor(np.arange(self.num_frames) * num_available / self.num_frames).astype(np.int64)

    def _as_clip_frames(self, view) -> List[Image.Image]:
        if isinstance(view, Image.Image):
            frames = [view]
        elif isinstance(view, (list, tuple)):
            frames = list(view)
            if not frames:
                raise ValueError("V-JEPA received an empty image clip.")
            if not all(isinstance(frame, Image.Image) for frame in frames):
                raise TypeError("V-JEPA clips must be lists/tuples of PIL images.")
        else:
            raise TypeError(f"Unsupported V-JEPA view type: {type(view)}")
        indices = self._frame_indices(len(frames))
        return [frames[int(idx)].convert("RGB") for idx in indices]

    def _view_to_video(self, view) -> np.ndarray:
        frames = []
        for image in self._as_clip_frames(view):
            if self.image_size:
                image = image.resize((self.image_size, self.image_size))
            frames.append(np.asarray(image))
        return np.stack(frames, axis=0)

    def _view_to_torchhub_video(self, view) -> torch.Tensor:
        frames = [self.image_transform(image.convert("RGB")) for image in self._as_clip_frames(view)]
        return torch.stack(frames, dim=1)  # [C, T, H, W]

    def _encode_flat_views_torchhub(self, flat_views: List) -> torch.Tensor:
        self._ensure_encoder_compute_dtype()
        pixel_values = torch.stack(
            [self._view_to_torchhub_video(view) for view in flat_views],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            tokens = self.encoder(pixel_values)
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[-1]
        return tokens.detach()

    def _encode_flat_views_hf(self, flat_views: List) -> torch.Tensor:
        if not flat_views:
            raise ValueError("FrozenVJEPA2Encoder received no images.")

        pixel_batches = []
        for view in flat_views:
            video = self._view_to_video(view)
            inputs = self.processor(videos=video, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(device=self.device, dtype=self.dtype)
            pixel_batches.append(pixel_values)

        pixel_values = torch.cat(pixel_batches, dim=0)
        with torch.no_grad():
            if hasattr(self.encoder, "get_vision_features"):
                tokens = self.encoder.get_vision_features(pixel_values_videos=pixel_values)
            else:
                try:
                    outputs = self.encoder(
                        pixel_values_videos=pixel_values,
                        skip_predictor=True,
                        return_dict=True,
                    )
                except TypeError:
                    outputs = self.encoder(pixel_values_videos=pixel_values, return_dict=True)
                tokens = getattr(outputs, "last_hidden_state", None)
                if tokens is None:
                    tokens = outputs[0]
        return tokens.detach()

    def _compress_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() > 3:
            tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        if tokens.dim() != 3:
            raise ValueError(f"Unexpected V-JEPA token shape: {tuple(tokens.shape)}")
        if self.max_tokens <= 0 or tokens.shape[1] <= self.max_tokens:
            return tokens

        temporal_blocks = max(1, int(np.ceil(float(self.num_frames) / max(1, self.tubelet_size))))
        if tokens.shape[1] % temporal_blocks == 0 and self.max_tokens >= temporal_blocks:
            spatial_tokens = tokens.shape[1] // temporal_blocks
            per_block = max(1, self.max_tokens // temporal_blocks)
            remainder = self.max_tokens - per_block * temporal_blocks
            tokens_by_time = tokens.reshape(
                tokens.shape[0],
                temporal_blocks,
                spatial_tokens,
                tokens.shape[-1],
            )
            pooled_blocks = []
            for block_idx in range(temporal_blocks):
                block_tokens = tokens_by_time[:, block_idx]
                block_target = per_block + (1 if block_idx < remainder else 0)
                pooled = F.adaptive_avg_pool1d(
                    block_tokens.transpose(1, 2).float(),
                    output_size=block_target,
                ).transpose(1, 2)
                pooled_blocks.append(pooled.to(tokens.dtype))
            return torch.cat(pooled_blocks, dim=1)

        pooled = F.adaptive_avg_pool1d(
            tokens.transpose(1, 2).float(),
            output_size=self.max_tokens,
        ).transpose(1, 2)
        return pooled.to(tokens.dtype)

    def _encode_flat_views(self, flat_views: List[Image.Image]) -> torch.Tensor:
        if not flat_views:
            raise ValueError("FrozenVJEPA2Encoder received no images.")
        if self.backend == "torchhub":
            tokens = self._encode_flat_views_torchhub(flat_views)
        else:
            tokens = self._encode_flat_views_hf(flat_views)
        return self._compress_tokens(tokens)

    def forward(self, batch_views: List[List[Image.Image]]) -> torch.Tensor:
        """
        Args:
            batch_views: List of samples; each sample is a list of PIL images.

        Returns:
            Tensor [B, view_tokens, hidden_size], concatenating tokens across views.
        """
        counts = [len(views) for views in batch_views]
        if any(count == 0 for count in counts):
            raise ValueError("Each sample must contain at least one image view.")

        flat_views = [view for views in batch_views for view in views]
        flat_tokens = self._encode_flat_views(flat_views)

        per_sample = []
        cursor = 0
        for count in counts:
            tokens = flat_tokens[cursor : cursor + count]
            cursor += count
            per_sample.append(tokens.reshape(1, -1, tokens.shape[-1]))

        return torch.cat(per_sample, dim=0)
