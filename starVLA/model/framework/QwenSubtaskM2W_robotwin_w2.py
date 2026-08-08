"""
Robotwin W2 framework.

RoboTwin has two wrist cameras. This variant keeps V-JEPA encoding per camera
but merges left/right latent tokens inside each latent time slot before one JEPA
predictor forward pass.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.QwenSubtaskM2W_w2 import Qwen_SubtaskM2W, _cfg_bool
from starVLA.model.modules.world_model import VisionTransformerPredictorAC
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch


logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenSubtaskM2W_robotwin_w2")
class Qwen_SubtaskM2W_RobotwinW2(Qwen_SubtaskM2W):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        jepa_cfg = self._jepa_predictor_config(config)
        self.num_wrist_views = int(
            jepa_cfg.get(
                "num_wrist_views",
                config.datasets.vla_data.get("num_wrist_views", 2),
            )
        )
        if self.num_wrist_views != 2:
            raise ValueError(
                "QwenSubtaskM2W_robotwin_w2 expects exactly two wrist views "
                f"(left/right), got num_wrist_views={self.num_wrist_views}."
            )
        if self.jepa_prediction_view != "wrist":
            raise ValueError(
                "QwenSubtaskM2W_robotwin_w2 only supports "
                "framework.jepa_predictor.prediction_view=wrist."
            )
        self._rebuild_multiview_jepa_predictor(config, jepa_cfg)
        logger.info(
            "Initialized Robotwin W2 view-aware JEPA | "
            f"num_wrist_views={self.num_wrist_views} | "
            f"latent_frames={self.jepa_latent_frames} | "
            f"grid={self.jepa_grid_size}x{self.jepa_grid_size * self.num_wrist_views}"
        )

    def _split_main_wrist_views(self, example: dict):
        """Split Robotwin online observations as [head, left_wrist, right_wrist]."""
        image = example["image"]
        explicit_wrist = example.get("wrist_views", None)

        if explicit_wrist is not None:
            main_image = image[0] if isinstance(image, (list, tuple)) else image
            return to_pil_preserve([main_image]), to_pil_preserve(explicit_wrist)

        if isinstance(image, (list, tuple)) and len(image) >= self.num_wrist_views + 1:
            return (
                to_pil_preserve([image[0]]),
                to_pil_preserve(list(image[1 : 1 + self.num_wrist_views])),
            )

        raise ValueError(
            "QwenSubtaskM2W_robotwin_w2 expects Robotwin observations as "
            f"[head, left_wrist, right_wrist], got {len(image) if isinstance(image, (list, tuple)) else 1} image(s)."
        )

    def _rebuild_multiview_jepa_predictor(self, config, jepa_cfg) -> None:
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        jepa_predictor_dim = self.visual_encoder.hidden_size
        jepa_num_heads = int(jepa_cfg.get("jepa_predictor_num_heads", 8))
        if jepa_predictor_dim % jepa_num_heads != 0:
            raise ValueError(
                "V-JEPA hidden size must be divisible by "
                f"jepa_predictor_num_heads, got {jepa_predictor_dim} and {jepa_num_heads}."
            )

        grid_h = self.jepa_grid_size
        grid_w = self.jepa_grid_size * self.num_wrist_views
        self.jepa_predictor = VisionTransformerPredictorAC(
            img_size=(grid_h * self.jepa_patch_size, grid_w * self.jepa_patch_size),
            patch_size=self.jepa_patch_size,
            num_frames=self.jepa_latent_frames,
            tubelet_size=1,
            embed_dim=self.visual_encoder.hidden_size,
            predictor_embed_dim=jepa_predictor_dim,
            depth=int(jepa_cfg.get("jepa_predictor_depth", 4)),
            num_heads=jepa_num_heads,
            action_embed_dim=hidden_size,
            is_frame_causal=False,
            use_bf16_autocast=_cfg_bool(
                jepa_cfg.get("jepa_predictor_bf16_autocast", True),
                True,
            ),
            use_activation_checkpointing=_cfg_bool(
                jepa_cfg.get("jepa_predictor_activation_checkpointing", False),
                False,
            ),
            drop_rate=float(jepa_cfg.get("dropout", 0.0)),
        )
        view_embeddings = torch.empty(self.num_wrist_views, self.visual_encoder.hidden_size)
        nn.init.normal_(view_embeddings, std=float(jepa_cfg.get("view_embed_init_std", 0.02)))
        self.jepa_predictor.view_embeddings = nn.Parameter(view_embeddings)

    def _merge_wrist_view_tokens(
        self,
        tokens: torch.Tensor,
        *,
        add_view_embedding: bool,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected V-JEPA tokens [B, N, D], got {tuple(tokens.shape)}.")
        batch_size, num_tokens, hidden_dim = tokens.shape
        spatial_tokens = self.jepa_grid_size * self.jepa_grid_size
        expected_tokens = self.num_wrist_views * self.jepa_latent_frames * spatial_tokens
        if num_tokens != expected_tokens:
            raise ValueError(
                "Robotwin W2 expected concatenated left/right wrist tokens with "
                f"N={expected_tokens} (=views {self.num_wrist_views} * latent_frames "
                f"{self.jepa_latent_frames} * spatial_tokens {spatial_tokens}), "
                f"got N={num_tokens}."
            )
        if hidden_dim != self.visual_encoder.hidden_size:
            raise ValueError(
                f"Expected V-JEPA hidden dim {self.visual_encoder.hidden_size}, got {hidden_dim}."
            )

        tokens = tokens.reshape(
            batch_size,
            self.num_wrist_views,
            self.jepa_latent_frames,
            spatial_tokens,
            hidden_dim,
        )
        if add_view_embedding:
            view_embeddings = self.jepa_predictor.view_embeddings.to(
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + view_embeddings.view(1, self.num_wrist_views, 1, 1, hidden_dim)

        tokens = tokens.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            self.jepa_latent_frames,
            self.num_wrist_views * spatial_tokens,
            hidden_dim,
        )
        return tokens.flatten(1, 2).contiguous()

    def _predict_jepa_future_tokens(
        self,
        h_reason: torch.Tensor,
        current_views,
    ) -> torch.Tensor:
        predictor = self.jepa_predictor
        predictor_dtype = predictor.predictor_embed.weight.dtype
        with torch.no_grad():
            current_tokens = self.visual_encoder(current_views).to(
                h_reason.device,
                dtype=predictor_dtype,
            )
        current_tokens = self._merge_wrist_view_tokens(
            current_tokens,
            add_view_embedding=True,
        )
        action_tokens = self._jepa_action_tokens(
            h_reason,
            predictor_dtype,
            self.jepa_latent_frames,
        )
        return predictor(current_tokens, action_tokens)

    def _jepa_future_loss(
        self,
        h_reason: torch.Tensor,
        current_views,
        future_views,
        future_loss_weights=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_jepa_loss or future_views is None:
            return h_reason.new_tensor(0.0), None

        predictor_dtype = self.jepa_predictor.predictor_embed.weight.dtype
        pred_future = self._predict_jepa_future_tokens(
            h_reason=h_reason,
            current_views=current_views,
        )
        with torch.no_grad():
            future_tokens = self.visual_encoder(future_views).to(
                h_reason.device,
                dtype=predictor_dtype,
            )
            future_tokens = self._merge_wrist_view_tokens(
                future_tokens,
                add_view_embedding=False,
            )

        sample_weight = None
        if future_loss_weights is not None:
            sample_weight = torch.as_tensor(
                future_loss_weights,
                device=h_reason.device,
                dtype=torch.float32,
            )
        loss = self._weighted_l1_loss(
            pred_future,
            future_tokens,
            sample_weight=sample_weight,
        )
        return loss, pred_future
