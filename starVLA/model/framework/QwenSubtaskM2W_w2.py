"""
QwenSubtaskM2W framework.

Subtask-Guided Main-to-Wrist VLA:
  - Supervise and generate explicit M2W CoT text.
  - Frozen V-JEPA2.1 encodes the current wrist frame and a future wrist clip.
  - JEPA predictor uses Qwen reasoning as action tokens to predict future wrist tokens.
  - A JEPA latent loss regularizes future-token prediction.
  - The existing DiT flow-matching action head predicts action chunks.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.frozen_visual_encoder import FrozenVJEPA2Encoder
from starVLA.model.modules.world_model import VisionTransformerPredictorAC
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images


logger = initialize_overwatch(__name__)


def _cfg_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class WristFutureContextBlock(nn.Module):
    """Extract action-relevant context from dense JEPA wrist tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_context = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention compresses future wrist latents without changing Qwen tokens.
        q = self.norm_query(query)
        kv = self.norm_context(context)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        query = query + self.dropout(attn_out)
        query = query + self.dropout(self.mlp(self.norm_mlp(query)))
        return query


class WristFutureContextAdapter(nn.Module):
    """Resample dense future-wrist JEPA tokens into compact action context."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_context_tokens: int = 32,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if num_context_tokens <= 0:
            raise ValueError(f"num_context_tokens must be positive, got {num_context_tokens}.")
        if input_dim % num_heads != 0:
            raise ValueError(
                f"JEPA wrist context input_dim must be divisible by num_heads, "
                f"got {input_dim} and {num_heads}."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_context_tokens = int(num_context_tokens)
        self.input_norm = nn.LayerNorm(input_dim)
        # The query count determines the number of wrist context tokens.
        self.query_tokens = nn.Parameter(torch.randn(num_context_tokens, input_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                WristFutureContextBlock(
                    hidden_dim=input_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected wrist tokens [B, N, D], got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected wrist token dim {self.input_dim}, got {tokens.shape[-1]}."
            )

        adapter_dtype = self.query_tokens.dtype
        context = self.input_norm(tokens.to(dtype=adapter_dtype))
        # All samples share queries that cross-attend to their wrist tokens.
        query = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, context)
        return self.output_proj(self.output_norm(query))


@FRAMEWORK_REGISTRY.register("QwenSubtaskM2W_w2")
class Qwen_SubtaskM2W(Qwen_GR00T):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # The CoT prompt must come from the training or checkpoint config.
        self._require_reasoning_cot_prompt(config)
        self._sync_state_config(config)
        super().__init__(config=config, **kwargs)

        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        reasoning_cfg = self._reasoning_config(config)
        jepa_cfg = self._jepa_predictor_config(config)
        policy_cfg = config.framework.get("policy_context", {})
        vj_cfg = config.framework.get("vjepa2", {})
        self.cot_reasoning_mode = str(reasoning_cfg.get("cot_reasoning_mode", "explicit")).lower()
        if self.cot_reasoning_mode not in {"explicit", "implicit_query"}:
            raise ValueError(
                f"Unknown W2 cot_reasoning_mode={self.cot_reasoning_mode!r}; "
                "use `explicit` or `implicit_query`."
            )
        self.use_implicit_query_cot = self.cot_reasoning_mode == "implicit_query"
        self.qwen_action_context_mode = str(
            reasoning_cfg.get("qwen_action_context_mode", "full")
        ).lower()
        if self.qwen_action_context_mode not in {"full", "exclude_cot_prompt"}:
            raise ValueError(
                "Unknown qwen_action_context_mode="
                f"{self.qwen_action_context_mode!r}; use `full` or `exclude_cot_prompt`."
            )

        vjepa_input_frames = int(vj_cfg.get("num_frames", 8))
        vjepa_image_size = int(vj_cfg.get("image_size", 224))
        if vjepa_input_frames % 2 != 0:
            raise ValueError(f"W2 V-JEPA expects an even frame count, got {vjepa_input_frames}.")

        self.visual_encoder = FrozenVJEPA2Encoder(
            base_encoder=None,
            backend="torchhub",
            hub_repo=vj_cfg.get("hub_repo", "facebookresearch/vjepa2"),
            hub_model=vj_cfg.get("hub_model", "vjepa2_1_vit_large_384"),
            hub_source=vj_cfg.get("hub_source", "github"),
            pretrained=True,
            num_frames=vjepa_input_frames,
            image_size=vjepa_image_size,
            max_tokens=None,
        )

        self.lambda_cot = float(reasoning_cfg.get("lambda_cot", 0.1))
        self.lambda_jepa = float(jepa_cfg.get("lambda_jepa", 0.1))
        self.use_jepa_loss = _cfg_bool(jepa_cfg.get("use_jepa_loss", True), True)
        self.jepa_prediction_view = str(jepa_cfg.get("prediction_view", "wrist")).strip().lower()
        if self.jepa_prediction_view not in {"wrist", "main"}:
            raise ValueError(
                "Unknown framework.jepa_predictor.prediction_view="
                f"{self.jepa_prediction_view!r}; use `wrist` or `main`."
            )
        self.use_state = _cfg_bool(policy_cfg.get("use_state", False), False)
        self.vlm_include_wrist_view = _cfg_bool(
            config.datasets.vla_data.get("vlm_include_wrist_view", False), False
        )
        self.jepa_latent_frames = int(vjepa_input_frames // 2)
        self.generate_cot_at_inference = _cfg_bool(
            reasoning_cfg.get("generate_cot_at_inference", True), True
        )
        self.inference_cot_max_new_tokens = int(reasoning_cfg.get("inference_cot_max_new_tokens", 96))
        self.inference_cot_use_cache = _cfg_bool(
            reasoning_cfg.get("inference_cot_use_cache", True), True
        )
        self.inference_empty_cache = _cfg_bool(
            reasoning_cfg.get("inference_empty_cache", False), False
        )
        self.detach_wrist_action_context = _cfg_bool(
            policy_cfg.get("detach_wrist_action_context", True), True
        )

        self.latent_cot_query_tokens = []
        if self.use_implicit_query_cot:
            self._initialize_latent_cot_tokens(
                int(reasoning_cfg.get("latent_cot_num_queries", 16))
            )
            # Implicit-query inference consumes contextualized query states and
            # does not autoregressively emit textual CoT.
            self.generate_cot_at_inference = False

        # V-JEPA2.1 tubelet encoding maps eight wrist frames to four latent slots.
        # Predictor grid and patch settings must match the frozen token layout.
        jepa_predictor_dim = self.visual_encoder.hidden_size
        jepa_patch_size = 16
        if vjepa_image_size % jepa_patch_size != 0:
            raise ValueError(
                f"W2 V-JEPA image_size={vjepa_image_size} must be divisible by patch_size={jepa_patch_size}."
            )
        jepa_grid_size = vjepa_image_size // jepa_patch_size
        self.jepa_patch_size = jepa_patch_size
        self.jepa_grid_size = jepa_grid_size
        jepa_num_frames = self.jepa_latent_frames
        jepa_num_heads = int(jepa_cfg.get("jepa_predictor_num_heads", 8))
        if jepa_predictor_dim % jepa_num_heads != 0:
            raise ValueError(
                "V-JEPA hidden size must be divisible by "
                f"jepa_predictor_num_heads, got {jepa_predictor_dim} and {jepa_num_heads}."
            )
        self.jepa_predictor = VisionTransformerPredictorAC(
            img_size=(jepa_grid_size * jepa_patch_size, jepa_grid_size * jepa_patch_size),
            patch_size=jepa_patch_size,
            num_frames=jepa_num_frames,
            tubelet_size=1,
            embed_dim=self.visual_encoder.hidden_size,
            predictor_embed_dim=jepa_predictor_dim,
            depth=int(jepa_cfg.get("jepa_predictor_depth", 4)),
            num_heads=jepa_num_heads,
            action_embed_dim=hidden_size,
            is_frame_causal=False,
            use_bf16_autocast=_cfg_bool(
                jepa_cfg.get("jepa_predictor_bf16_autocast", True), True
            ),
            use_activation_checkpointing=_cfg_bool(
                jepa_cfg.get("jepa_predictor_activation_checkpointing", False), False
            ),
            dropout=jepa_cfg.get("dropout", 0.0),
        )
        self.wrist_context_adapter = WristFutureContextAdapter(
            input_dim=self.visual_encoder.hidden_size,
            output_dim=hidden_size,
            num_context_tokens=int(policy_cfg.get("wrist_action_context_tokens", 32)),
            num_heads=int(policy_cfg.get("wrist_action_context_heads", 8)),
            num_layers=int(policy_cfg.get("wrist_action_context_layers", 1)),
            dropout=float(
                policy_cfg.get(
                    "wrist_action_context_dropout",
                    jepa_cfg.get("dropout", 0.0),
                )
            ),
            mlp_ratio=float(policy_cfg.get("wrist_action_context_mlp_ratio", 2.0)),
        )

        if not self.use_state and getattr(self.action_model, "state_encoder", None) is not None:
            # Freeze the state encoder when proprioception is disabled.
            for param in self.action_model.state_encoder.parameters():
                param.requires_grad = False
            self.action_model.state_encoder.eval()
            # Remove the unused module so DDP does not track inactive parameters.
            self.action_model.state_encoder = None

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.eval()

        logger.info(
            f"[QwenSubtaskM2W] hidden={hidden_size} | "
            f"wrist_vjepa_dim={self.visual_encoder.hidden_size} | "
            f"lambda_cot={self.lambda_cot} | lambda_jepa={self.lambda_jepa} | "
            f"cot_reasoning_mode={self.cot_reasoning_mode} | "
            f"generate_cot_at_inference={self.generate_cot_at_inference} | "
            f"use_jepa_loss={self.use_jepa_loss} | "
            f"jepa_prediction_view={self.jepa_prediction_view} | "
            f"vlm_include_wrist_view={self.vlm_include_wrist_view} | "
            f"qwen_action_context_mode={self.qwen_action_context_mode} | "
            f"detach_wrist_action_context={self.detach_wrist_action_context} | "
            f"use_state={self.use_state} | "
            f"jepa_latent_frames={self.jepa_latent_frames} | "
            f"jepa_predictor_dim={jepa_predictor_dim} | "
            f"jepa_grid_size={jepa_grid_size} | "
            f"jepa_num_frames={jepa_num_frames} | "
            f"wrist_action_context_tokens="
            f"{getattr(self.wrist_context_adapter, 'num_context_tokens', 0)}"
        )

    def _initialize_latent_cot_tokens(self, num_queries: int) -> None:
        if num_queries <= 0:
            raise ValueError("latent_cot_num_queries must be positive")
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        tokens = [f"<|m2w_cot_query_{idx}|>" for idx in range(num_queries)]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        if any(token_id == tokenizer.unk_token_id for token_id in token_ids):
            raise RuntimeError("Failed to register latent CoT query tokens")
        self.latent_cot_query_tokens = tokens
        self.register_buffer(
            "latent_cot_query_ids",
            torch.tensor(token_ids, dtype=torch.long),
            persistent=True,
        )

    @staticmethod
    def _reasoning_config(config):
        framework_cfg = config.framework
        return framework_cfg.get("reasoning", {})

    @staticmethod
    def _jepa_predictor_config(config):
        framework_cfg = config.framework
        return framework_cfg.get("jepa_predictor", {})

    @staticmethod
    def _sync_state_config(config) -> None:
        """Keep policy_context.use_state consistent with action_model.state_dim."""
        if config is None:
            return
        try:
            policy_cfg = config.framework.get("policy_context", {})
            action_cfg = config.framework.action_model
        except Exception:
            return

        use_state = bool(policy_cfg.get("use_state", False))
        if not use_state:
            # Do not create a state encoder when proprioception is disabled;
            # otherwise DDP would see trainable parameters unused by forward.
            try:
                action_cfg.state_dim = 0
            except Exception:
                action_cfg["state_dim"] = 0
            return

        state_dim = int(action_cfg.get("state_dim", 0))
        if state_dim <= 0:
            raise ValueError(
                "framework.policy_context.use_state=true requires "
                "framework.action_model.state_dim > 0."
            )

    @staticmethod
    def _require_reasoning_cot_prompt(config) -> None:
        """Require CoT prompt to be supplied by YAML/checkpoint config."""
        if config is None:
            return
        try:
            vla_data = config.datasets.vla_data
        except Exception as exc:
            raise ValueError(
                "QwenSubtaskM2W requires datasets.vla_data.CoT_prompt in the "
                "training YAML or checkpoint config."
            ) from exc

        try:
            prompt = str(vla_data.get("CoT_prompt", "") or "")
        except Exception:
            prompt = str(getattr(vla_data, "CoT_prompt", "") or "")

        if not prompt.strip():
            raise ValueError(
                "QwenSubtaskM2W requires a non-empty "
                "datasets.vla_data.CoT_prompt. No built-in CoT prompt fallback "
                "is used."
            )

    def _split_main_wrist_views(self, example: dict):
        """
        Training samples usually provide explicit wrist_views. Online inference
        may pass [main, wrist] through the image field, so normalize both forms.
        """
        image = example["image"]
        explicit_wrist = example.get("wrist_views", None)

        if explicit_wrist is not None:
            if isinstance(image, (list, tuple)) and len(image) >= 2:
                return to_pil_preserve([image[0]]), to_pil_preserve(explicit_wrist)
            return to_pil_preserve(image), to_pil_preserve(explicit_wrist)

        if isinstance(image, (list, tuple)) and len(image) >= 2:
            return to_pil_preserve([image[0]]), to_pil_preserve([image[1]])

        return to_pil_preserve(image), to_pil_preserve(image)

    @staticmethod
    def _ensure_image_list(images) -> list:
        if isinstance(images, tuple):
            return list(images)
        if isinstance(images, list):
            return images
        return [images]

    @classmethod
    def _current_wrist_images_for_vlm(cls, wrist_views) -> list:
        current_images = []
        for view in cls._ensure_image_list(wrist_views):
            # Training wrist views are history clips; Qwen/VLM should see the
            # current wrist frame, which is the last frame in the clip.
            if isinstance(view, (list, tuple)):
                if not view:
                    continue
                view = view[-1]
            current_images.extend(cls._ensure_image_list(view))
        return current_images

    def _vlm_images(self, main_images, wrist_views, has_wrist_view: bool = True) -> list:
        images = list(self._ensure_image_list(main_images))
        if self.vlm_include_wrist_view and has_wrist_view:
            images.extend(self._current_wrist_images_for_vlm(wrist_views))
        return images

    def align_model_input(self, examples: List[dict]):
        # Align training and inference: Qwen sees configured views while the
        # prediction branch independently selects main or wrist observations.
        split_views = [self._split_main_wrist_views(example) for example in examples]
        main_images = [views[0] for views in split_views]
        wrist_views = [views[1] for views in split_views]
        batch_images = []
        for example, main, wrist in zip(examples, main_images, wrist_views):
            qwen_images = example.get("qwen_images", None)
            if qwen_images is not None:
                qwen_images = self._ensure_image_list(to_pil_preserve(qwen_images))
                if qwen_images:
                    batch_images.append(qwen_images)
                    continue
            batch_images.append(
                self._vlm_images(
                    main,
                    wrist,
                    has_wrist_view=bool(example.get("has_wrist_view", True)),
                )
            )
        main_views = [
            to_pil_preserve(example.get("main_views", main))
            for example, main in zip(examples, main_images)
        ]
        future_main_views = [
            to_pil_preserve(example.get("future_main_views", main))
            for example, main in zip(examples, main_views)
        ]
        future_wrist_views = [
            to_pil_preserve(example.get("future_wrist_views", views[1]))
            for example, views in zip(examples, split_views)
        ]
        instructions = [example["lang"] for example in examples]
        cot_targets = [example.get("cot_target", "") for example in examples]
        future_loss_weights = self._future_loss_weights(examples)
        if self.use_state:
            if "state" not in examples[0]:
                raise ValueError(
                    "framework.policy_context.use_state=true but dataset examples "
                    "do not contain a `state` field. Set datasets.vla_data.include_state=true."
                )
            state = [example["state"] for example in examples]
        else:
            state = None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", [224, 224])
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        return (
            batch_images,
            main_views,
            future_main_views,
            wrist_views,
            future_wrist_views,
            instructions,
            cot_targets,
            state,
            future_loss_weights,
        )

    @staticmethod
    def _future_loss_weights(examples: List[dict]) -> Optional[List[float]]:
        """Read dataset-provided JEPA weights or use an unweighted mean."""
        weights = []
        saw_future_weight = False
        for example in examples:
            if "jepa_loss_weight" not in example:
                weights.append(1.0)
                continue
            saw_future_weight = True
            try:
                weights.append(float(example.get("jepa_loss_weight", 1.0)))
            except (TypeError, ValueError):
                weights.append(1.0)
        return weights if saw_future_weight else None

    def _state_tensor(self, state, device, dtype):
        if state is None:
            return None
        return torch.from_numpy(np.array(state)).to(device=device, dtype=dtype)

    @staticmethod
    def _find_last_token_span(sequence: torch.Tensor, pattern: torch.Tensor) -> Optional[tuple[int, int]]:
        if pattern.numel() == 0 or pattern.numel() > sequence.numel():
            return None
        width = int(pattern.numel())
        for start in range(int(sequence.numel()) - width, -1, -1):
            if torch.equal(sequence[start : start + width], pattern):
                return start, start + width
        return None

    def _instruction_token_span(
        self,
        sequence: torch.Tensor,
        instruction: str,
        prompt_text: Optional[str] = None,
    ) -> Optional[tuple[int, int]]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer

        # Token boundaries depend on surrounding text. For example, Qwen can
        # encode an instruction-final period plus the following prompt newlines
        # as one token. Locate the exact rendered prompt first, then map the
        # instruction character range onto that prompt's token offsets.
        if prompt_text:
            instruction_start = prompt_text.find(instruction)
            if instruction_start >= 0:
                encoded_prompt = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                prompt_ids = torch.tensor(
                    encoded_prompt["input_ids"],
                    device=sequence.device,
                    dtype=sequence.dtype,
                )
                prompt_span = self._find_last_token_span(sequence, prompt_ids)
                if prompt_span is not None:
                    instruction_end = instruction_start + len(instruction)
                    relative_indices = [
                        index
                        for index, (start, end) in enumerate(
                            encoded_prompt["offset_mapping"]
                        )
                        if end > instruction_start and start < instruction_end
                    ]
                    if relative_indices:
                        return (
                            prompt_span[0] + relative_indices[0],
                            prompt_span[0] + relative_indices[-1] + 1,
                        )

        candidates = (
            f"Your task is: {instruction}",
            f"\nYour task is: {instruction}",
            f"Your task is {instruction}",
            f"\nYour task is {instruction}",
            instruction,
            f" {instruction}",
        )
        for text in candidates:
            ids = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.squeeze(0).to(sequence.device)
            span = self._find_last_token_span(sequence, ids)
            if span is not None:
                return span
        return None

    def _action_context_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        instructions: List[str],
        cot_mask: Optional[torch.Tensor] = None,
        prompt_texts: Optional[List[str]] = None,
    ) -> Optional[torch.Tensor]:
        if self.qwen_action_context_mode == "full":
            return None

        keep = torch.zeros_like(input_ids, dtype=torch.bool)
        model_config = self.qwen_vl_interface.model.config
        visual_token_ids = {
            int(token_id)
            for token_id in (
                getattr(model_config, "image_token_id", None),
                getattr(model_config, "video_token_id", None),
                getattr(model_config, "vision_start_token_id", None),
                getattr(model_config, "vision_end_token_id", None),
            )
            if token_id is not None
        }
        for token_id in visual_token_ids:
            keep |= input_ids.eq(token_id)

        for row, instruction in enumerate(instructions):
            prompt_text = prompt_texts[row] if prompt_texts is not None else None
            span = self._instruction_token_span(
                input_ids[row],
                str(instruction),
                prompt_text=prompt_text,
            )
            if span is None:
                raise RuntimeError(
                    "Unable to locate the raw instruction inside the Qwen CoT prompt; "
                    f"instruction={instruction!r}."
                )
            keep[row, span[0] : span[1]] = True

        if cot_mask is not None:
            keep |= cot_mask.to(device=keep.device, dtype=torch.bool)
        if attention_mask is not None:
            keep &= attention_mask.to(device=keep.device, dtype=torch.bool)
        return keep

    def _qwen_forward(
        self,
        batch_images,
        instructions,
        cot_targets=None,
    ):
        if cot_targets is None or any(not str(target).strip() for target in cot_targets):
            raise ValueError("CoT training requires a non-empty cot_target for every sample.")
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=cot_targets,
            label_mode="assistant",
            return_solution_mask=True,
            use_cot_prompt=True,
        )
        solution_mask = qwen_inputs.pop("solution_mask", None)
        if solution_mask is None:
            raise RuntimeError("Qwen input builder did not return solution_mask for CoT training.")
        action_context_mask = self._action_context_mask(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs.get("attention_mask"),
            instructions=instructions,
            cot_mask=solution_mask,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
        hidden = outputs.hidden_states[connect_layer_index]
        if outputs.loss is None:
            raise RuntimeError("Qwen did not return a CoT loss.")
        cot_loss = outputs.loss

        mask = solution_mask.to(hidden.device)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
        h_reason = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom

        if action_context_mask is not None:
            action_context_mask = action_context_mask.to(hidden.device, dtype=torch.bool)
        return hidden, h_reason, cot_loss, action_context_mask

    def _qwen_encode_instruction(self, batch_images, instructions):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=None,
            use_cot_prompt=False,
        )
        action_context_mask = self._action_context_mask(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs.get("attention_mask"),
            instructions=instructions,
            cot_mask=None,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
        hidden = outputs.hidden_states[connect_layer_index]
        h_reason = hidden.mean(dim=1)
        if action_context_mask is not None:
            action_context_mask = action_context_mask.to(hidden.device, dtype=torch.bool)
        return hidden, h_reason, action_context_mask

    def _implicit_query_prompts(self, instructions: List[str]) -> List[str]:
        if not self.latent_cot_query_tokens:
            raise RuntimeError("Implicit CoT mode is enabled but query tokens are not initialized")

        query_suffix = " ".join(self.latent_cot_query_tokens)
        try:
            cot_prompt = str(self.config.datasets.vla_data.get("CoT_prompt", "") or "")
        except Exception:
            cot_prompt = str(getattr(self.config.datasets.vla_data, "CoT_prompt", "") or "")

        prompts = []
        for instruction in instructions:
            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt.strip() else instruction
            prompts.append(f"{prompt.rstrip()}\n{query_suffix}")
        return prompts

    def _implicit_query_inputs(
        self,
        batch_images,
        instructions,
        cot_targets=None,
    ):
        prompt_texts = self._implicit_query_prompts(instructions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=prompt_texts,
            solutions=cot_targets,
            label_mode="assistant",
            use_cot_prompt=False,
        )
        input_ids = qwen_inputs["input_ids"]
        query_ids = self.latent_cot_query_ids.to(input_ids.device)
        query_mask = (input_ids.unsqueeze(-1) == query_ids.view(1, 1, -1)).any(dim=-1)
        expected = int(query_ids.numel())
        counts = query_mask.sum(dim=1)
        if not torch.all(counts == expected):
            raise RuntimeError(
                "Each sample must contain every implicit CoT query token exactly once; "
                f"expected={expected}, counts={counts.tolist()}"
            )

        action_context_mask = self._action_context_mask(
            input_ids=input_ids,
            attention_mask=qwen_inputs.get("attention_mask"),
            instructions=instructions,
            cot_mask=query_mask,
            prompt_texts=prompt_texts,
        )
        return qwen_inputs, query_mask, expected, action_context_mask

    def _qwen_forward_query_hidden(self, qwen_inputs, query_mask, expected: int):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
        hidden = outputs.hidden_states[connect_layer_index]
        query_hidden = hidden[query_mask.to(hidden.device)].view(
            hidden.shape[0],
            expected,
            hidden.shape[-1],
        )
        return hidden, query_hidden, outputs

    def _qwen_forward_implicit_query_cot(
        self,
        batch_images,
        instructions,
        cot_targets=None,
    ):
        if cot_targets is None or any(not str(target).strip() for target in cot_targets):
            raise ValueError(
                "Implicit CoT query training requires a non-empty cot_target "
                "for every sample."
            )
        qwen_inputs, query_mask, expected, action_context_mask = self._implicit_query_inputs(
            batch_images,
            instructions,
            cot_targets=cot_targets,
        )
        hidden, h_reason, outputs = self._qwen_forward_query_hidden(
            qwen_inputs,
            query_mask,
            expected,
        )
        if action_context_mask is not None:
            action_context_mask = action_context_mask.to(hidden.device, dtype=torch.bool)
        return hidden, h_reason, outputs.loss, action_context_mask

    def _qwen_encode_implicit_query(self, batch_images, instructions):
        qwen_inputs, query_mask, expected, action_context_mask = self._implicit_query_inputs(
            batch_images,
            instructions=instructions,
            cot_targets=None,
        )
        hidden, h_reason, _ = self._qwen_forward_query_hidden(
            qwen_inputs,
            query_mask,
            expected,
        )
        if action_context_mask is not None:
            action_context_mask = action_context_mask.to(hidden.device, dtype=torch.bool)
        return hidden, h_reason, action_context_mask

    @staticmethod
    def _clean_generated_cot(text: str) -> str:
        text = str(text or "").strip()
        for stop in ("<|im_end|>", "<|endoftext|>"):
            if stop in text:
                text = text.split(stop, 1)[0].strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("text"):
                text = text[4:].strip()
        return " ".join(text.split())

    def _decode_generated_cot(self, generated_ids: torch.Tensor) -> List[str]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [self._clean_generated_cot(text) for text in texts]

    def _generated_token_mask(self, generated_ids: torch.Tensor, length: int) -> torch.Tensor:
        mask = torch.ones(generated_ids.shape[0], length, device=generated_ids.device, dtype=torch.bool)
        token_len = min(length, generated_ids.shape[1])
        token_slice = generated_ids[:, :token_len]
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        special_ids = {
            idx for idx in (
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                getattr(tokenizer, "bos_token_id", None),
            )
            if idx is not None
        }
        if special_ids:
            for token_id in special_ids:
                mask[:, :token_len] &= token_slice != token_id
        return mask

    def _generation_hidden_to_reason(
        self,
        generation_hidden,
        input_len: int,
        generated_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
        prompt_hidden = None
        longest_hidden = None
        generated_pieces = []

        if not generation_hidden:
            raise RuntimeError("Qwen generation did not return hidden states.")

        # Transformers versions expose generation hidden states either by
        # decoding step or directly by layer; support both structures.
        direct_layers = generation_hidden
        if len(direct_layers) > 0 and torch.is_tensor(direct_layers[0]):
            hidden = direct_layers[connect_layer_index]
            reason_tokens = hidden[:, input_len:, :] if hidden.shape[1] > input_len else hidden
            prompt_len = min(input_len, hidden.shape[1])
            return hidden, reason_tokens.mean(dim=1), prompt_len

        for step_hidden in generation_hidden:
            if step_hidden is None:
                continue
            hidden = step_hidden[connect_layer_index] if isinstance(step_hidden, (tuple, list)) else step_hidden
            if hidden is None or hidden.dim() != 3:
                continue

            if longest_hidden is None or hidden.shape[1] > longest_hidden.shape[1]:
                longest_hidden = hidden
            if hidden.shape[1] >= input_len:
                prompt_hidden = hidden[:, :input_len, :]
            if hidden.shape[1] > input_len:
                generated_pieces = [hidden[:, input_len:, :]]
            elif hidden.shape[1] == 1:
                generated_pieces.append(hidden)

        if generated_pieces:
            generated_hidden = torch.cat(generated_pieces, dim=1)
            if prompt_hidden is None:
                mask = self._generated_token_mask(generated_ids, generated_hidden.shape[1]).to(generated_hidden.device)
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(generated_hidden.dtype)
                h_reason = (generated_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
                return generated_hidden, h_reason, 0
            full_hidden = torch.cat([prompt_hidden, generated_hidden], dim=1)
            mask = self._generated_token_mask(generated_ids, generated_hidden.shape[1]).to(generated_hidden.device)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(generated_hidden.dtype)
            h_reason = (generated_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            return full_hidden, h_reason, min(input_len, full_hidden.shape[1])

        if prompt_hidden is None:
            prompt_hidden = longest_hidden
        if prompt_hidden is None:
            raise RuntimeError("Unable to recover Qwen hidden states from generation output.")

        return prompt_hidden, prompt_hidden.mean(dim=1), min(input_len, prompt_hidden.shape[1])

    def _generate_cot_with_hidden(
        self,
        batch_images,
        instructions,
    ) -> tuple[torch.Tensor, torch.Tensor, List[str], Optional[torch.Tensor]]:
        # Preserve generated-token hidden states when explicit CoT is requested.
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=None,
        )
        input_len = qwen_inputs["input_ids"].shape[1]
        prompt_action_mask = self._action_context_mask(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs.get("attention_mask"),
            instructions=instructions,
            cot_mask=None,
        )
        generated = self.qwen_vl_interface.generate(
            **qwen_inputs,
            max_new_tokens=self.inference_cot_max_new_tokens,
            do_sample=False,
            use_cache=self.inference_cot_use_cache,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        generated_ids = sequences[:, input_len:]
        texts = self._decode_generated_cot(generated_ids)
        hidden, h_reason, hidden_prompt_len = self._generation_hidden_to_reason(
            generation_hidden=getattr(generated, "hidden_states", None),
            input_len=input_len,
            generated_ids=generated_ids,
        )
        action_context_mask = None
        if prompt_action_mask is not None:
            action_context_mask = torch.zeros(
                hidden.shape[:2],
                device=hidden.device,
                dtype=torch.bool,
            )
            if hidden_prompt_len > 0:
                action_context_mask[:, :hidden_prompt_len] = prompt_action_mask[
                    :, :hidden_prompt_len
                ].to(hidden.device)
            gen_start = hidden_prompt_len
            gen_len = hidden.shape[1] - gen_start
            if gen_len > 0:
                generated_valid_mask = self._generated_token_mask(
                    generated_ids,
                    gen_len,
                ).to(hidden.device)
                action_context_mask[:, gen_start : gen_start + gen_len] = (
                    generated_valid_mask[:, :gen_len]
                )
        return hidden, h_reason, texts, action_context_mask

    def _jepa_action_tokens(
        self,
        h_reason: torch.Tensor,
        dtype: torch.dtype,
        num_frames: int,
    ) -> torch.Tensor:
        # h_reason contains implicit-query states [B, 16, D], repeated per JEPA slot.
        num_frames = max(1, int(num_frames))
        return (
            h_reason.to(dtype)
            .unsqueeze(1)
            .expand(-1, num_frames, -1, -1)
            .flatten(1, 2)
            .contiguous()
        )

    @staticmethod
    def _weighted_l1_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_sample = F.l1_loss(
            pred.float(),
            target.detach().float(),
            reduction="none",
        ).mean(dim=tuple(range(1, pred.dim())))
        if sample_weight is None:
            return per_sample.mean()
        weight = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).view(-1)
        denom = weight.sum().clamp(min=1.0)
        return (per_sample * weight).sum() / denom

    def _predict_jepa_future_tokens(
        self,
        h_reason: torch.Tensor,
        current_views,
    ) -> torch.Tensor:
        # Frozen V-JEPA encodes history clips; gradients update only the predictor.
        predictor = self.jepa_predictor
        predictor_dtype = predictor.predictor_embed.weight.dtype
        with torch.no_grad():
            current_tokens = self.visual_encoder(current_views).to(
                h_reason.device,
                dtype=predictor_dtype,
            )

        latent_frames = self.jepa_latent_frames
        action_tokens = self._jepa_action_tokens(h_reason, predictor_dtype, latent_frames)
        return predictor(current_tokens, action_tokens)

    def _jepa_future_loss(
        self,
        h_reason: torch.Tensor,
        current_views,
        future_views,
        future_loss_weights=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # L_jepa supervises future latents rather than RGB reconstruction.
        if not self.use_jepa_loss or future_views is None:
            return h_reason.new_tensor(0.0), None

        predictor = self.jepa_predictor
        predictor_dtype = predictor.predictor_embed.weight.dtype
        pred_future = self._predict_jepa_future_tokens(
            h_reason=h_reason,
            current_views=current_views,
        )
        with torch.no_grad():
            future_tokens = self.visual_encoder(future_views).to(
                h_reason.device,
                dtype=predictor_dtype,
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

    def _append_wrist_action_context(
        self,
        hidden: torch.Tensor,
        pred_future_tokens: torch.Tensor,
        action_context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Compress predicted future latents into action-context tokens.
        wrist_tokens = pred_future_tokens
        if self.detach_wrist_action_context:
            wrist_tokens = wrist_tokens.detach()

        wrist_context = self.wrist_context_adapter(wrist_tokens)
        wrist_context = wrist_context.to(device=hidden.device, dtype=hidden.dtype)
        hidden = torch.cat([hidden, wrist_context], dim=1)
        if action_context_mask is None:
            return hidden, None
        wrist_mask = torch.ones(
            wrist_context.shape[:2],
            device=action_context_mask.device,
            dtype=torch.bool,
        )
        action_context_mask = torch.cat(
            [action_context_mask.to(torch.bool), wrist_mask],
            dim=1,
        )
        return hidden, action_context_mask

    def get_action_condition(
        self,
        batch_images,
        main_views,
        future_main_views,
        wrist_views,
        future_wrist_views,
        instructions,
        cot_targets=None,
        state=None,
        future_loss_weights=None,
    ):
        # Build action conditioning from Qwen states, future context, and optional state.
        if self.use_implicit_query_cot:
            hidden, h_reason, cot_loss, action_context_mask = self._qwen_forward_implicit_query_cot(
                batch_images=batch_images,
                instructions=instructions,
                cot_targets=cot_targets,
            )
        else:
            hidden, h_reason, cot_loss, action_context_mask = self._qwen_forward(
                batch_images=batch_images,
                instructions=instructions,
                cot_targets=cot_targets,
            )
        if self.jepa_prediction_view == "main":
            current_views = main_views
            future_views = future_main_views
        else:
            current_views = wrist_views
            future_views = future_wrist_views
        jepa_loss, pred_future_tokens = self._jepa_future_loss(
            h_reason=h_reason,
            current_views=current_views,
            future_views=future_views,
            future_loss_weights=future_loss_weights,
        )
        if pred_future_tokens is None:
            pred_future_tokens = self._predict_jepa_future_tokens(
                h_reason=h_reason,
                current_views=current_views,
            )
        state_tensor = self._state_tensor(state, hidden.device, hidden.dtype)
        return (
            hidden,
            state_tensor,
            cot_loss,
            jepa_loss,
            pred_future_tokens,
            action_context_mask,
        )

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        (
            batch_images,
            main_views,
            future_main_views,
            wrist_views,
            future_wrist_views,
            instructions,
            cot_targets,
            state,
            future_loss_weights,
        ) = self.align_model_input(examples)

        (
            last_hidden,
            state,
            cot_loss,
            jepa_loss,
            pred_future,
            action_context_mask,
        ) = self.get_action_condition(
            batch_images=batch_images,
            main_views=main_views,
            future_main_views=future_main_views,
            wrist_views=wrist_views,
            future_wrist_views=future_wrist_views,
            instructions=instructions,
            cot_targets=cot_targets,
            state=state,
            future_loss_weights=future_loss_weights,
        )
        last_hidden, action_context_mask = self._append_wrist_action_context(
            last_hidden,
            pred_future,
            action_context_mask=action_context_mask,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array([example["action"] for example in examples]),
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )
            actions_target = actions[:, -self.action_horizon:, :]

            repeated = self.config.trainer.get("repeated_diffusion_steps", 4)
            actions_rep = actions_target.repeat(repeated, 1, 1)
            hidden_rep = last_hidden.repeat(repeated, 1, 1)
            state_rep = state.repeat(repeated, 1, 1) if state is not None else None
            action_mask_rep = (
                action_context_mask.repeat(repeated, 1)
                if action_context_mask is not None
                else None
            )
            action_loss = self.action_model(
                hidden_rep,
                actions_rep,
                state_rep,
                encoder_attention_mask=action_mask_rep,
            )

        cot_loss_weighted = self.lambda_cot * cot_loss
        jepa_loss_weighted = self.lambda_jepa * jepa_loss
        total_loss = action_loss + cot_loss_weighted + jepa_loss_weighted
        return {
            "action_loss": action_loss,
            "cot_loss_weighted": cot_loss_weighted,
            "jepa_loss_weighted": jepa_loss_weighted,
            "lambda_cot": hidden_rep.new_tensor(self.lambda_cot),
            "lambda_jepa": hidden_rep.new_tensor(self.lambda_jepa),
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        (
            batch_images,
            main_views,
            _,
            wrist_views,
            _,
            instructions,
            _,
            state,
            _,
        ) = self.align_model_input(examples)

        if self.generate_cot_at_inference:
            hidden, h_reason, generated_cot, action_context_mask = self._generate_cot_with_hidden(
                batch_images,
                instructions,
            )
            if self.inference_empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Fast evaluation reuses prompt encoding without CoT generation or teacher forcing.
            if self.use_implicit_query_cot:
                hidden, h_reason, action_context_mask = self._qwen_encode_implicit_query(
                    batch_images=batch_images,
                    instructions=instructions,
                )
            else:
                hidden, h_reason, action_context_mask = self._qwen_encode_instruction(
                    batch_images=batch_images,
                    instructions=instructions,
                )
            generated_cot = [""] * len(instructions)
        state = self._state_tensor(state, hidden.device, hidden.dtype)
        current_views = main_views if self.jepa_prediction_view == "main" else wrist_views
        pred_future_tokens = self._predict_jepa_future_tokens(
            h_reason=h_reason,
            current_views=current_views,
        )
        hidden, action_context_mask = self._append_wrist_action_context(
            hidden,
            pred_future_tokens,
            action_context_mask=action_context_mask,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self.action_model.predict_action(
                hidden,
                state,
                encoder_attention_mask=action_context_mask,
            )

        result = {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
        result["generated_cot"] = generated_cot
        if pred_future_tokens is not None:
            result["pred_future_tokens"] = pred_future_tokens.detach().float().cpu().numpy()
        return result
