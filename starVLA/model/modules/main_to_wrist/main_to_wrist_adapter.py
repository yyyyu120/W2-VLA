"""
Subtask-guided main-to-wrist adapter.

Qwen3-VL provides main-view reasoning. Frozen V-JEPA2.1 is used only on
wrist-view images; this module turns Qwen reasoning into compact wrist latent
queries and reads task-conditioned View 2 tokens from wrist dense tokens.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureWristTokenMixer(nn.Module):
    """Lightweight token-mixing predictor for compact future wrist latents."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if bottleneck_dim % num_heads != 0:
            raise ValueError("future predictor bottleneck_dim must be divisible by num_heads")
        self.input_proj = nn.Linear(hidden_dim * 2, bottleneck_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=num_heads,
            dim_feedforward=bottleneck_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_proj = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(
        self,
        c_wrist: torch.Tensor,
        q_wrist: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self.input_proj.weight.dtype
        x = self.input_proj(torch.cat([c_wrist, q_wrist], dim=-1).to(dtype))
        delta = self.output_proj(self.token_mixer(x))
        return c_wrist.to(delta.dtype) + delta


class MainToWristAdapter(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        hidden_dim: int,
        state_dim: int = 0,
        num_heads: int = 8,
        dropout: float = 0.0,
        num_latent_tokens: int = 16,
        query_condition_mode: str = "reason_qwen",
        future_predictor_type: str = "mlp",
        future_predictor_bottleneck_dim: int = 512,
        future_predictor_num_layers: int = 2,
        future_predictor_num_heads: int = 8,
        use_ema_target_projector: bool = False,
        target_ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.state_dim = int(state_dim or 0)
        self.num_latent_tokens = int(num_latent_tokens or 1)
        self.query_condition_mode = str(query_condition_mode or "reason_qwen").lower()
        self.future_predictor_type = str(future_predictor_type or "mlp").lower()
        self.use_ema_target_projector = bool(use_ema_target_projector)
        self.target_ema_decay = float(target_ema_decay)

        self.wrist_proj = nn.Linear(visual_dim, hidden_dim)

        self.wrist_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * self.num_latent_tokens),
        )
        if self.future_predictor_type == "mlp":
            self.future_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.future_token_mixer = None
        elif self.future_predictor_type in {"token_mixer", "transformer", "transformer_2l"}:
            self.future_predictor = None
            self.future_token_mixer = FutureWristTokenMixer(
                hidden_dim=hidden_dim,
                bottleneck_dim=int(future_predictor_bottleneck_dim),
                num_layers=int(future_predictor_num_layers),
                num_heads=int(future_predictor_num_heads),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown future_predictor_type={self.future_predictor_type!r}")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.wrist_norm = nn.LayerNorm(hidden_dim)
        self.future_norm = nn.LayerNorm(hidden_dim)

        if self.use_ema_target_projector:
            self.target_wrist_proj = copy.deepcopy(self.wrist_proj)
            self.target_wrist_cross_attn = copy.deepcopy(self.wrist_cross_attn)
            self.target_future_norm = copy.deepcopy(self.future_norm)
            for module in self._target_modules():
                module.requires_grad_(False)
        else:
            self.target_wrist_proj = None
            self.target_wrist_cross_attn = None
            self.target_future_norm = None

    def _target_modules(self):
        return (
            self.target_wrist_proj,
            self.target_wrist_cross_attn,
            self.target_future_norm,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_ema_target_projector:
            for module in self._target_modules():
                module.eval()
        return self

    @torch.no_grad()
    def update_target_ema(self) -> None:
        if not self.use_ema_target_projector:
            return
        online_modules = (self.wrist_proj, self.wrist_cross_attn, self.future_norm)
        for target_module, online_module in zip(self._target_modules(), online_modules):
            for target_param, online_param in zip(
                target_module.parameters(), online_module.parameters()
            ):
                target_param.mul_(self.target_ema_decay).add_(
                    online_param.detach(), alpha=1.0 - self.target_ema_decay
                )
            for target_buffer, online_buffer in zip(
                target_module.buffers(), online_module.buffers()
            ):
                target_buffer.copy_(online_buffer)

    def _cross_attend(
        self,
        attn: nn.MultiheadAttention,
        query: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        squeeze = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze = True
        out, _ = attn(query=query, key=tokens, value=tokens)
        return out.squeeze(1) if squeeze else out

    def forward(
        self,
        wrist_tokens: torch.Tensor,
        reasoning_state: torch.Tensor,
        qwen_hidden: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        dtype = self.wrist_proj.weight.dtype
        wrist_tokens = self.wrist_proj(wrist_tokens.to(dtype))
        reasoning_state = reasoning_state.to(dtype)

        if self.query_condition_mode in {"latent_cot", "latent_query"}:
            if reasoning_state.dim() != 3:
                raise ValueError(
                    "latent_cot query mode expects reasoning_state shaped [B, N, H], "
                    f"got {tuple(reasoning_state.shape)}"
                )
            q_wrist = self.query_norm(reasoning_state)
            c_wrist = self._cross_attend(self.wrist_cross_attn, q_wrist, wrist_tokens)
            c_wrist = self.wrist_norm(c_wrist)
            return {
                "h_reason": reasoning_state,
                "qwen_context": reasoning_state.mean(dim=1),
                "q_wrist": q_wrist,
                "c_wrist": c_wrist,
                "z_wrist": c_wrist,
            }

        if self.query_condition_mode in {"reason_only", "cot_only", "h_reason"}:
            qwen_context = reasoning_state
        elif qwen_hidden is not None:
            qwen_context = qwen_hidden.to(dtype).mean(dim=1)
        else:
            qwen_context = reasoning_state

        batch_size = reasoning_state.shape[0]
        q_wrist = self.query_mlp(
            torch.cat(
                [
                    reasoning_state,
                    qwen_context,
                ],
                dim=-1,
            ).to(dtype)
        ).view(batch_size, self.num_latent_tokens, self.hidden_dim)
        q_wrist = self.query_norm(q_wrist)

        c_wrist = self._cross_attend(self.wrist_cross_attn, q_wrist, wrist_tokens)
        c_wrist = self.wrist_norm(c_wrist)

        return {
            "h_reason": reasoning_state,
            "qwen_context": qwen_context,
            "q_wrist": q_wrist,
            "c_wrist": c_wrist,
            "z_wrist": c_wrist,
        }

    def extract_future_target(
        self,
        future_wrist_tokens: torch.Tensor,
        q_wrist: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_ema_target_projector:
            dtype = self.target_wrist_proj.weight.dtype
            tokens = self.target_wrist_proj(future_wrist_tokens.to(dtype))
            target = self._cross_attend(
                self.target_wrist_cross_attn,
                q_wrist.detach().to(dtype),
                tokens,
            )
            return self.target_future_norm(target)
        dtype = self.wrist_proj.weight.dtype
        tokens = self.wrist_proj(future_wrist_tokens.to(dtype))
        target = self._cross_attend(self.wrist_cross_attn, q_wrist.detach().to(dtype), tokens)
        return self.future_norm(target)

    def predict_future(
        self,
        c_wrist: torch.Tensor,
        q_wrist: torch.Tensor,
    ) -> torch.Tensor:
        if self.future_token_mixer is not None:
            pred = self.future_token_mixer(
                c_wrist=c_wrist,
                q_wrist=q_wrist,
            )
        else:
            dtype = self.future_predictor[0].weight.dtype
            pred = self.future_predictor(
                torch.cat([c_wrist.to(dtype), q_wrist.to(dtype)], dim=-1)
            )
        return self.future_norm(pred)

    @staticmethod
    def future_latent_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_sample = F.smooth_l1_loss(
            pred.float(),
            target.detach().float(),
            reduction="none",
        ).mean(dim=tuple(range(1, pred.dim())))
        if sample_weight is None:
            return per_sample.mean()
        weight = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).view(-1)
        denom = weight.sum().clamp(min=1.0)
        return (per_sample * weight).sum() / denom

    @staticmethod
    def future_state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return MainToWristAdapter.future_latent_loss(pred, target)
