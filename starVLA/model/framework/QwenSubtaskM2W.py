"""
QwenSubtaskM2W framework.

Subtask-Guided Main-to-Wrist VLA:
  - Historical modes supervise and generate explicit M2W CoT text.
  - V8-style modes use parallel latent CoT query states, with no textual
    decoding at inference.
  - Frozen V-JEPA2.1 encodes current and future wrist views only.
  - Main-to-wrist adapter sends Qwen-derived latent queries into wrist dense tokens.
  - Wrist history clips may be passed to V-JEPA2.1 so the wrist branch can use
    short-term motion cues instead of repeated single-frame pseudo-video.
  - A wrist-local future-latent loss regularizes compact future wrist tokens.
  - The existing DiT flow-matching action head predicts action chunks.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.QwenGR00T import IGNORE_INDEX, Qwen_GR00T
from starVLA.model.modules.frozen_visual_encoder import FrozenVJEPA2Encoder
from starVLA.model.modules.main_to_wrist import MainToWristAdapter
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


_M2W_COT_PROMPT_LIBERO_DUAL_V3_COMPACT = """Your task is: {instruction}

Using the current main-view and wrist-view images, identify the robot's current manipulation step.

Output exactly:
Subtask: ...
Reasoning: ...
Wrist: ...

Do not add extra text."""


_M2W_COT_PROMPT_ROBOTWIN_TRI_V1 = """Your task is: {instruction}

Using the current high camera, left wrist camera, and right wrist camera images, identify the robot's current manipulation step.

Output exactly:
Subtask: ...
Reasoning: ...
Wrist: ...

Do not add extra text."""


_STARVLA_BBOX_PROMPT_V1 = "Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format."


# Keep legacy names as aliases so old configs with only `cot_prompt_version`
# still load.  Checkpoints whose config.yaml already contains an explicit
# CoT_prompt bypass this registry and keep their exact train-time prompt.
_M2W_COT_PROMPTS = {
    "libero_dual_v3_compact": _M2W_COT_PROMPT_LIBERO_DUAL_V3_COMPACT,
    "robotwin_tri_v1": _M2W_COT_PROMPT_ROBOTWIN_TRI_V1,
    "starvla_bbox_v1": _STARVLA_BBOX_PROMPT_V1,
    "libero_legacy_v1": _M2W_COT_PROMPT_LIBERO_DUAL_V3_COMPACT,
    "libero_dual_v2": _M2W_COT_PROMPT_LIBERO_DUAL_V3_COMPACT,
    "libero_robot_wrist_v4": _M2W_COT_PROMPT_LIBERO_DUAL_V3_COMPACT,
}


@FRAMEWORK_REGISTRY.register("QwenSubtaskM2W")
class Qwen_SubtaskM2W(Qwen_GR00T):
    _COT_EXTRA_FIELD_MARKERS = (
        "\nNote:",
        " Note:",
        "Note:",
        "\nGrounded Fields:",
        " Grounded Fields:",
        "Grounded Fields:",
        "\nAction:",
        " Action:",
        "Action:",
        "\nGrounding:",
        " Grounding:",
        "Grounding:",
        "\nTarget:",
        " Target:",
        "Target:",
        "\nWhere:",
        " Where:",
        "Where:",
        "\nPhase:",
        " Phase:",
        "Phase:",
        "\nProgress:",
        " Progress:",
        "Progress:",
        "\nNextTrend:",
        " NextTrend:",
        "NextTrend:",
    )

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        self._ensure_reasoning_cot_prompt(config)
        super().__init__(config=config, **kwargs)

        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        state_dim = int(config.framework.action_model.get("state_dim", 0) or 0)
        m2w_cfg = config.framework.get("main_to_wrist", {})
        vj_cfg = config.framework.get("vjepa2", {})
        self.cot_reasoning_mode = str(m2w_cfg.get("cot_reasoning_mode", "explicit")).lower()
        query_cot_modes = {
            "latent_query",
            "parallel_query",
            "implicit_query",
            "qwen_implicit",
            "implicit_cot",
        }
        latent_cot_num_queries = int(m2w_cfg.get("latent_cot_num_queries", 8))
        wrist_query_count = (
            latent_cot_num_queries
            if self.cot_reasoning_mode in query_cot_modes
            else int(m2w_cfg.get("num_latent_tokens", 16))
        )

        self.lambda_cot = float(m2w_cfg.get("lambda_cot", 0.1))
        self.lambda_wrist = float(m2w_cfg.get("lambda_wrist", 0.1))
        self.use_cot_loss = _cfg_bool(m2w_cfg.get("use_cot_loss", True), True)
        self.use_wrist_future_loss = _cfg_bool(
            m2w_cfg.get("use_wrist_future_loss", True), True
        )
        self.generate_cot_at_inference = _cfg_bool(
            m2w_cfg.get("generate_cot_at_inference", True), True
        )
        self.inference_cot_max_new_tokens = int(m2w_cfg.get("inference_cot_max_new_tokens", 96))
        self.inference_cot_use_cache = _cfg_bool(
            m2w_cfg.get("inference_cot_use_cache", True), True
        )
        self.inference_empty_cache = _cfg_bool(
            m2w_cfg.get("inference_empty_cache", False), False
        )
        try:
            prompt_version = str(config.datasets.vla_data.get("cot_prompt_version", "") or "")
        except Exception:
            prompt_version = str(getattr(config.datasets.vla_data, "cot_prompt_version", "") or "")
        self.use_prompt_without_cot_loss = _cfg_bool(
            m2w_cfg.get("use_prompt_without_cot_loss", prompt_version == "starvla_bbox_v1"),
            prompt_version == "starvla_bbox_v1",
        )
        self.action_condition_mode = str(m2w_cfg.get("action_condition_mode", "concat")).lower()
        self.query_condition_mode = str(m2w_cfg.get("query_condition_mode", "reason_qwen")).lower()
        self.qwen_input_views = str(m2w_cfg.get("qwen_input_views", "main")).lower()
        self.qwen_action_context_mode = str(
            m2w_cfg.get("qwen_action_context_mode", "full")
        ).lower()
        if self.qwen_action_context_mode not in {"full", "exclude_cot_prompt"}:
            raise ValueError(
                "Unknown qwen_action_context_mode="
                f"{self.qwen_action_context_mode!r}; use `full` or `exclude_cot_prompt`."
            )
        self.include_wrist_query_in_action = _cfg_bool(
            m2w_cfg.get("include_wrist_query_in_action", False), False
        )
        self.include_future_wrist_in_action = _cfg_bool(
            m2w_cfg.get("include_future_wrist_in_action", True), True
        )

        self.skip_vjepa_init = _cfg_bool(
            m2w_cfg.get("skip_vjepa_init", False), False
        ) or (self.action_condition_mode == "qwen_only" and not self.use_wrist_future_loss)
        if self.skip_vjepa_init and self.action_condition_mode != "qwen_only":
            raise ValueError(
                "skip_vjepa_init=true is only valid when action_condition_mode='qwen_only'. "
                f"Got action_condition_mode={self.action_condition_mode!r}."
            )

        self.visual_encoder = None
        self.m2w_adapter = None
        if not self.skip_vjepa_init:
            self.visual_encoder = FrozenVJEPA2Encoder(
                base_encoder=vj_cfg.get("base_encoder", None),
                backend=vj_cfg.get("backend", "auto"),
                hub_repo=vj_cfg.get("hub_repo", "facebookresearch/vjepa2"),
                hub_model=vj_cfg.get("hub_model", "vjepa2_1_vit_large_384"),
                hub_source=vj_cfg.get("hub_source", "github"),
                pretrained=vj_cfg.get("pretrained", True),
                num_frames=vj_cfg.get("num_frames", 16),
                image_size=vj_cfg.get("image_size", 384),
                max_tokens=vj_cfg.get("max_tokens", 256),
            )
            self.m2w_adapter = MainToWristAdapter(
                visual_dim=self.visual_encoder.hidden_size,
                hidden_dim=hidden_size,
                state_dim=state_dim,
                num_heads=m2w_cfg.get("num_heads", 8),
                dropout=m2w_cfg.get("dropout", 0.0),
                num_latent_tokens=wrist_query_count,
                query_condition_mode=m2w_cfg.get("query_condition_mode", "reason_qwen"),
                future_predictor_type=m2w_cfg.get("future_predictor_type", "mlp"),
                future_predictor_bottleneck_dim=m2w_cfg.get(
                    "future_predictor_bottleneck_dim", 512
                ),
                future_predictor_num_layers=m2w_cfg.get("future_predictor_num_layers", 2),
                future_predictor_num_heads=m2w_cfg.get("future_predictor_num_heads", 8),
                use_ema_target_projector=_cfg_bool(
                    m2w_cfg.get("use_ema_target_projector", False), False
                ),
                target_ema_decay=m2w_cfg.get("target_ema_decay", 0.99),
            )

        self.latent_cot_query_tokens = []
        if self.cot_reasoning_mode in query_cot_modes:
            self._initialize_latent_cot_tokens(latent_cot_num_queries)
            # Query-token CoT modes never autoregressively emit text at policy
            # inference.  They use the contextualized query hidden states.
            self.generate_cot_at_inference = False

        if self.action_condition_mode == "qwen_only" and self.m2w_adapter is not None:
            for param in self.m2w_adapter.parameters():
                param.requires_grad = False

        if self.visual_encoder is not None:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()

        wrist_vjepa_dim = (
            "skipped" if self.visual_encoder is None else self.visual_encoder.hidden_size
        )
        m2w_latent_tokens = (
            "skipped" if self.m2w_adapter is None else self.m2w_adapter.num_latent_tokens
        )
        future_predictor_type = (
            "skipped" if self.m2w_adapter is None else self.m2w_adapter.future_predictor_type
        )
        ema_target = (
            "skipped" if self.m2w_adapter is None else self.m2w_adapter.use_ema_target_projector
        )

        logger.info(
            f"[QwenSubtaskM2W] hidden={hidden_size} | "
            f"wrist_vjepa_dim={wrist_vjepa_dim} | "
            f"m2w_latent_tokens={m2w_latent_tokens} | "
            f"lambda_cot={self.lambda_cot} | lambda_wrist={self.lambda_wrist} | "
            f"cot_reasoning_mode={self.cot_reasoning_mode} | "
            f"generate_cot_at_inference={self.generate_cot_at_inference} | "
            f"action_condition_mode={self.action_condition_mode} | "
            f"query_condition_mode={self.query_condition_mode} | "
            f"future_predictor_type={future_predictor_type} | "
            f"ema_target={ema_target} | "
            f"skip_vjepa_init={self.skip_vjepa_init} | "
            f"qwen_input_views={self.qwen_input_views} | "
            f"qwen_action_context_mode={self.qwen_action_context_mode} | "
            f"include_wrist_query_in_action={self.include_wrist_query_in_action} | "
            f"include_future_wrist_in_action={self.include_future_wrist_in_action}"
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
    def _ensure_reasoning_cot_prompt(config) -> None:
        """Preserve checkpoint prompts and fill only genuinely missing prompts.

        Checkpoint configs contain the exact prompt used during training.  That
        text is authoritative at inference, including for legacy runs.  Prompt
        versions are therefore metadata/fallbacks rather than migration rules.
        """
        if config is None:
            return
        try:
            vla_data = config.datasets.vla_data
        except Exception:
            return

        try:
            prompt = str(vla_data.get("CoT_prompt", "") or "")
        except Exception:
            prompt = str(getattr(vla_data, "CoT_prompt", "") or "")

        # Never rewrite an explicit prompt loaded from a checkpoint. This is
        # what guarantees train/eval prompt identity for both old and new runs.
        if prompt.strip():
            return

        default_version = "libero_dual_v3_compact"
        try:
            version = str(vla_data.get("cot_prompt_version", default_version) or default_version)
        except Exception:
            version = str(getattr(vla_data, "cot_prompt_version", default_version) or default_version)
        if version not in _M2W_COT_PROMPTS:
            raise ValueError(
                f"Unknown cot_prompt_version={version!r}; "
                f"available versions: {sorted(_M2W_COT_PROMPTS)}"
            )
        selected_prompt = _M2W_COT_PROMPTS[version]
        try:
            vla_data.CoT_prompt = selected_prompt
        except Exception:
            vla_data["CoT_prompt"] = selected_prompt

    def _split_main_wrist_views(self, example: dict):
        """
        Training samples provide separate image / wrist_views fields. Legacy
        server evaluation samples may provide [main, wrist] in image only; split that
        layout here so M2W keeps Qwen on the main view and V-JEPA on the wrist view.
        """
        image = example["image"]
        explicit_wrist = example.get("wrist_views", None)

        if explicit_wrist is not None:
            if isinstance(image, (list, tuple)) and len(image) >= 2:
                return to_pil_preserve([image[0]]), to_pil_preserve(explicit_wrist)
            return to_pil_preserve(image), to_pil_preserve(explicit_wrist)

        if isinstance(image, (list, tuple)) and len(image) >= 2:
            return to_pil_preserve([image[0]]), to_pil_preserve(list(image[1:]))

        return to_pil_preserve(image), to_pil_preserve(image)

    @staticmethod
    def _current_wrist_frame(wrist_views):
        """Pick the current wrist frame from a wrist view or history clip.

        Training/eval M2W samples store wrist history as [camera][time]. Qwen
        should receive the current wrist image only, matching the base dual-view
        VLA input instead of seeing the full temporal clip.
        """
        if isinstance(wrist_views, Image.Image):
            return wrist_views
        if isinstance(wrist_views, tuple):
            wrist_views = list(wrist_views)
        if not isinstance(wrist_views, list) or not wrist_views:
            return wrist_views
        first = wrist_views[0]
        if isinstance(first, Image.Image):
            return first
        if isinstance(first, tuple):
            first = list(first)
        if isinstance(first, list) and first:
            return first[-1]
        return first

    @classmethod
    def _current_wrist_frames(cls, wrist_views):
        """Return all current wrist camera frames from single or multi-wrist input."""
        if isinstance(wrist_views, Image.Image):
            return [wrist_views]
        if isinstance(wrist_views, tuple):
            wrist_views = list(wrist_views)
        if not isinstance(wrist_views, list) or not wrist_views:
            return [wrist_views]
        if all(isinstance(view, Image.Image) for view in wrist_views):
            return list(wrist_views)
        frames = []
        for view in wrist_views:
            frames.append(cls._current_wrist_frame(view))
        return frames

    def _qwen_images_from_views(self, main_images, wrist_views):
        if self.qwen_input_views in {"main", "main_only"}:
            return main_images
        if self.qwen_input_views in {"dual", "main_wrist", "main+wrist"}:
            qwen_images = []
            for main, wrist in zip(main_images, wrist_views):
                main_list = main if isinstance(main, list) else [main]
                qwen_images.append(main_list + [self._current_wrist_frame(wrist)])
            return qwen_images
        if self.qwen_input_views in {
            "all",
            "tri",
            "tri_view",
            "main_all_wrist",
            "main+wrist_all",
        }:
            qwen_images = []
            for main, wrist in zip(main_images, wrist_views):
                main_list = main if isinstance(main, list) else [main]
                qwen_images.append(main_list + self._current_wrist_frames(wrist))
            return qwen_images
        raise ValueError(
            f"Unknown qwen_input_views={self.qwen_input_views!r}; use `main`, `dual`, or `all`."
        )

    def align_model_input(self, examples: List[dict]):
        split_views = [self._split_main_wrist_views(example) for example in examples]
        main_images = [views[0] for views in split_views]
        wrist_views = [views[1] for views in split_views]
        batch_images = self._qwen_images_from_views(main_images, wrist_views)
        future_wrist_views = [
            to_pil_preserve(example.get("future_wrist_views", views[1]))
            for example, views in zip(examples, split_views)
        ]
        instructions = [example["lang"] for example in examples]
        cot_targets = [example.get("cot_target", "") for example in examples]
        future_loss_weights = self._future_loss_weights(examples)
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", [224, 224])
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        return (
            batch_images,
            wrist_views,
            future_wrist_views,
            instructions,
            cot_targets,
            state,
            future_loss_weights,
        )

    @staticmethod
    def _future_loss_weights(examples: List[dict]) -> Optional[List[float]]:
        weights = []
        saw_future_weight = False
        for example in examples:
            if "future_wrist_loss_weight" not in example:
                weights.append(1.0)
                continue
            saw_future_weight = True
            try:
                weights.append(float(example.get("future_wrist_loss_weight", 1.0)))
            except (TypeError, ValueError):
                weights.append(1.0)
        return weights if saw_future_weight else None

    def _state_tensor(self, state, device, dtype):
        if state is None:
            return None
        return torch.from_numpy(np.array(state)).to(device=device, dtype=dtype)

    def _qwen_supports_logits_to_keep(self) -> bool:
        cached = getattr(self, "_cached_qwen_supports_logits_to_keep", None)
        if cached is not None:
            return cached
        import inspect

        try:
            supports = "logits_to_keep" in inspect.signature(self.qwen_vl_interface.model.forward).parameters
        except (TypeError, ValueError):
            supports = False
        self._cached_qwen_supports_logits_to_keep = supports
        return supports

    @staticmethod
    def _find_last_token_span(sequence: torch.Tensor, pattern: torch.Tensor) -> Optional[tuple[int, int]]:
        """Return the last exact occurrence of pattern in a 1-D token sequence."""
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
    ) -> Optional[tuple[int, int]]:
        """Locate the task instruction inside the longer CoT meta-prompt.

        The versioned prompts place the raw task after an ``Instruction:``
        marker.  Trying the marker form first prevents an object-name fragment
        elsewhere in the rules from being selected by mistake.
        """
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        candidates = (
            f"Instruction: {instruction}",
            f"\nInstruction: {instruction}",
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
    ) -> Optional[torch.Tensor]:
        """Select vision, raw instruction, and CoT tokens for the DiT.

        Qwen still receives the complete versioned CoT prompt for generation.
        Only the downstream action cross-attention excludes those meta-prompt
        rules.  ``full`` deliberately returns None to preserve historical
        checkpoint behavior exactly.
        """
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
            span = self._instruction_token_span(input_ids[row], str(instruction))
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

    def _qwen_forward(self, batch_images, instructions, cot_targets=None, compute_cot_loss: bool = True):
        has_cot = (
            cot_targets is not None
            and any(bool(str(target).strip()) for target in cot_targets)
        )
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=cot_targets if has_cot else None,
            label_mode="assistant",
            return_solution_mask=True,
            use_cot_prompt=bool(
                (has_cot and self.use_cot_loss and compute_cot_loss)
                or (not has_cot and self.use_prompt_without_cot_loss)
            ),
        )
        solution_mask = qwen_inputs.pop("solution_mask", None)
        action_context_mask = self._action_context_mask(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs.get("attention_mask"),
            instructions=instructions,
            cot_mask=solution_mask,
        )
        reason_labels = qwen_inputs.get("labels")
        if not compute_cot_loss:
            qwen_inputs.pop("labels", None)
            if self._qwen_supports_logits_to_keep():
                # We only need hidden states for action conditioning at inference.
                qwen_inputs["logits_to_keep"] = 1

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
        hidden = outputs.hidden_states[connect_layer_index]
        cot_loss = (
            outputs.loss
            if compute_cot_loss and self.use_cot_loss and has_cot and outputs.loss is not None
            else hidden.new_tensor(0.0)
        )

        if solution_mask is not None and solution_mask.any():
            mask = solution_mask.to(hidden.device)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
            h_reason = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            labels = reason_labels
            if labels is not None:
                mask = (labels != IGNORE_INDEX).to(hidden.device)
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
                h_reason = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                h_reason = hidden.mean(dim=1)

        return hidden, h_reason, cot_loss, action_context_mask

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
            if cot_prompt.strip():
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            prompts.append(f"{prompt.rstrip()}\n{query_suffix}")
        return prompts

    def _qwen_forward_implicit_query_cot(
        self,
        batch_images,
        instructions,
        cot_targets=None,
        compute_cot_loss: bool = True,
    ):
        """Qwen-native implicit CoT.

        Query tokens are inserted before the assistant CoT text.  During
        training, Qwen's own LM head predicts the teacher-forced CoT answer;
        during inference, only the contextualized query hidden states are used.
        The DiT action context excludes teacher-forced/generated CoT text.
        """
        has_cot = (
            cot_targets is not None
            and any(bool(str(target).strip()) for target in cot_targets)
            and self.use_cot_loss
            and compute_cot_loss
        )
        prompts = self._implicit_query_prompts(instructions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=prompts,
            solutions=cot_targets if has_cot else None,
            label_mode="assistant",
            return_solution_mask=True,
            use_cot_prompt=False,
        )
        qwen_inputs.pop("solution_mask", None)

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
        )

        if not has_cot:
            qwen_inputs.pop("labels", None)
            if self._qwen_supports_logits_to_keep():
                qwen_inputs["logits_to_keep"] = 1

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
        h_reason = query_hidden.mean(dim=1)
        cot_loss = (
            outputs.loss
            if has_cot and outputs.loss is not None
            else hidden.new_tensor(0.0)
        )
        if action_context_mask is not None:
            action_context_mask = action_context_mask.to(hidden.device, dtype=torch.bool)
        return hidden, h_reason, cot_loss, action_context_mask

    @classmethod
    def _clean_generated_cot(cls, text: str) -> str:
        text = str(text or "").strip()
        for stop in ("<|im_end|>", "<|endoftext|>"):
            if stop in text:
                text = text.split(stop, 1)[0].strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("text"):
                text = text[4:].strip()

        if "Subtask:" in text and "Reasoning:" in text and "Wrist:" in text:
            text = text[text.find("Subtask:") :]
            search_start = text.find("Wrist:") + len("Wrist:")
        else:
            search_start = 0

        cutoff = len(text)
        for marker in cls._COT_EXTRA_FIELD_MARKERS:
            pos = text.find(marker, search_start)
            if pos != -1:
                cutoff = min(cutoff, pos)
        text = text[:cutoff].strip()
        return " ".join(text.split())

    def _decode_generated_cot(self, generated_ids: torch.Tensor) -> List[str]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [self._clean_generated_cot(text) for text in texts]

    def _generation_stop_token_ids(self) -> List[int]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        token_ids = []
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ):
            if (
                token_id is not None
                and int(token_id) >= 0
                and int(token_id) != getattr(tokenizer, "unk_token_id", None)
                and int(token_id) not in token_ids
            ):
                token_ids.append(int(token_id))
        return token_ids

    @staticmethod
    def _find_subsequence(tokens: List[int], pattern: List[int], end: int) -> Optional[int]:
        if not pattern or end < len(pattern):
            return None
        last_start = end - len(pattern)
        for pos in range(last_start + 1):
            if tokens[pos : pos + len(pattern)] == pattern:
                return pos
        return None

    def _generated_token_cutoff_counts(self, generated_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        stop_ids = {
            token_id
            for token_id in (
                *self._generation_stop_token_ids(),
                tokenizer.pad_token_id,
                getattr(tokenizer, "bos_token_id", None),
            )
            if token_id is not None
        }
        marker_token_ids = []
        for marker in self._COT_EXTRA_FIELD_MARKERS:
            ids = tokenizer(marker, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
            marker_ids = [int(token_id) for token_id in ids.tolist()]
            if marker_ids and marker_ids not in marker_token_ids:
                marker_token_ids.append(marker_ids)

        counts = []
        for row in generated_ids:
            tokens = [int(token_id) for token_id in row.tolist()]
            cutoff = len(tokens)
            for idx, token_id in enumerate(tokens):
                if token_id in stop_ids:
                    cutoff = idx
                    break
            for marker_ids in marker_token_ids:
                pos = self._find_subsequence(tokens, marker_ids, cutoff)
                if pos is not None:
                    cutoff = min(cutoff, pos)
            counts.append(cutoff)
        return torch.tensor(counts, device=generated_ids.device, dtype=torch.long)

    def _generated_token_mask(
        self,
        generated_ids: torch.Tensor,
        length: int,
        cutoff_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cutoff_counts is None:
            cutoff_counts = self._generated_token_cutoff_counts(generated_ids)
        cutoff_counts = cutoff_counts.to(device=generated_ids.device).clamp(min=0, max=length)
        positions = torch.arange(length, device=generated_ids.device).unsqueeze(0)
        mask = positions < cutoff_counts.unsqueeze(1)
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
        if token_len < length:
            mask[:, token_len:] = False
        return mask

    def _generation_hidden_to_reason(
        self,
        generation_hidden,
        input_len: int,
        generated_ids: torch.Tensor,
        cutoff_counts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
        prompt_hidden = None
        longest_hidden = None
        generated_pieces = []

        if not generation_hidden:
            raise RuntimeError("Qwen generation did not return hidden states.")

        # HF generate returns either a tuple of decoding steps, each containing
        # layer hidden states, or a direct tuple of layer hidden states. Support
        # both shapes so the code is robust across transformers versions.
        direct_layers = generation_hidden
        if len(direct_layers) > 0 and torch.is_tensor(direct_layers[0]):
            hidden = direct_layers[connect_layer_index]
            prompt_len = min(input_len, hidden.shape[1])
            if hidden.shape[1] <= input_len:
                return hidden, hidden.mean(dim=1), prompt_len, None
            reason_tokens = hidden[:, input_len:, :]
            mask = self._generated_token_mask(
                generated_ids,
                reason_tokens.shape[1],
                cutoff_counts=cutoff_counts,
            ).to(reason_tokens.device)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(reason_tokens.dtype)
            h_reason = (reason_tokens * mask.unsqueeze(-1)).sum(dim=1) / denom
            return hidden, h_reason, prompt_len, mask

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
                mask = self._generated_token_mask(
                    generated_ids,
                    generated_hidden.shape[1],
                    cutoff_counts=cutoff_counts,
                ).to(generated_hidden.device)
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(generated_hidden.dtype)
                h_reason = (generated_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
                return generated_hidden, h_reason, 0, mask
            full_hidden = torch.cat([prompt_hidden, generated_hidden], dim=1)
            mask = self._generated_token_mask(
                generated_ids,
                generated_hidden.shape[1],
                cutoff_counts=cutoff_counts,
            ).to(generated_hidden.device)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(generated_hidden.dtype)
            h_reason = (generated_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            return full_hidden, h_reason, min(input_len, full_hidden.shape[1]), mask

        if prompt_hidden is None:
            prompt_hidden = longest_hidden
        if prompt_hidden is None:
            raise RuntimeError("Unable to recover Qwen hidden states from generation output.")

        return prompt_hidden, prompt_hidden.mean(dim=1), min(input_len, prompt_hidden.shape[1]), None

    def _generate_cot_with_hidden(
        self,
        batch_images,
        instructions,
    ) -> tuple[torch.Tensor, torch.Tensor, List[str], Optional[torch.Tensor]]:
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
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        stop_token_ids = self._generation_stop_token_ids()
        generation_kwargs = {
            "max_new_tokens": self.inference_cot_max_new_tokens,
            "do_sample": False,
            "use_cache": self.inference_cot_use_cache,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }
        if stop_token_ids:
            generation_kwargs["eos_token_id"] = stop_token_ids
            generation_kwargs["pad_token_id"] = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else stop_token_ids[0]
            )
        generated = self.qwen_vl_interface.generate(**qwen_inputs, **generation_kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        generated_ids = sequences[:, input_len:]
        cutoff_counts = self._generated_token_cutoff_counts(generated_ids)
        texts = self._decode_generated_cot(generated_ids)
        hidden, h_reason, hidden_prompt_len, generated_valid_mask = self._generation_hidden_to_reason(
            generation_hidden=getattr(generated, "hidden_states", None),
            input_len=input_len,
            generated_ids=generated_ids,
            cutoff_counts=cutoff_counts,
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
            if generated_valid_mask is not None:
                gen_start = hidden_prompt_len
                gen_len = min(generated_valid_mask.shape[1], hidden.shape[1] - gen_start)
                if gen_len > 0:
                    action_context_mask[:, gen_start : gen_start + gen_len] = (
                        generated_valid_mask[:, :gen_len].to(hidden.device)
                    )
        return hidden, h_reason, texts, action_context_mask

    def _condition_from_qwen_hidden(
        self,
        hidden: torch.Tensor,
        h_reason: torch.Tensor,
        wrist_views,
        future_wrist_views,
        state=None,
        future_loss_weights=None,
        action_context_mask: Optional[torch.Tensor] = None,
    ):
        if self.action_condition_mode == "qwen_only":
            state_tensor = self._state_tensor(state, hidden.device, hidden.dtype)
            return hidden, state_tensor, hidden.new_tensor(0.0), action_context_mask

        adapter_dtype = next(self.m2w_adapter.parameters()).dtype
        hidden_for_condition = hidden.to(adapter_dtype)
        h_reason = h_reason.to(adapter_dtype)
        state_tensor = self._state_tensor(state, hidden.device, adapter_dtype)

        if self.visual_encoder is None or self.m2w_adapter is None:
            raise RuntimeError(
                "M2W/V-JEPA branch is disabled by skip_vjepa_init, but "
                f"action_condition_mode={self.action_condition_mode!r} requires it."
            )

        with torch.no_grad():
            wrist_tokens = self.visual_encoder(wrist_views).to(hidden.device, dtype=adapter_dtype)

        adapter_out = self.m2w_adapter(
            wrist_tokens=wrist_tokens,
            reasoning_state=h_reason,
            qwen_hidden=hidden_for_condition,
            state=state_tensor,
        )
        pred_future = self.m2w_adapter.predict_future(
            c_wrist=adapter_out["c_wrist"],
            q_wrist=adapter_out["q_wrist"],
        )

        wrist_latent_loss = hidden.new_tensor(0.0)
        if self.use_wrist_future_loss and future_wrist_views is not None:
            with torch.no_grad():
                future_tokens = self.visual_encoder(future_wrist_views).to(hidden.device, dtype=adapter_dtype)
                target = self.m2w_adapter.extract_future_target(
                    future_wrist_tokens=future_tokens,
                    q_wrist=adapter_out["q_wrist"].detach(),
                )
            sample_weight = None
            if future_loss_weights is not None:
                future_weight = torch.as_tensor(
                    future_loss_weights,
                    device=hidden.device,
                    dtype=torch.float32,
                )
                sample_weight = future_weight if sample_weight is None else sample_weight * future_weight
            wrist_latent_loss = self.m2w_adapter.future_latent_loss(
                pred_future,
                target,
                sample_weight=sample_weight,
            )

        if self.action_condition_mode in {"qwen_future", "qwen_pred_future"}:
            # DiT receives the Qwen sequence plus the predicted future wrist
            # latent. Current wrist evidence C_wrist and Q_wrist are used to
            # form pred_future, but are not appended as separate action tokens.
            last_hidden = torch.cat(
                [hidden_for_condition, pred_future.to(hidden_for_condition.dtype)],
                dim=1,
            )
            if action_context_mask is not None:
                future_mask = torch.ones(
                    pred_future.shape[:2],
                    device=action_context_mask.device,
                    dtype=torch.bool,
                )
                action_context_mask = torch.cat(
                    [action_context_mask.to(torch.bool), future_mask],
                    dim=1,
                )
        elif self.action_condition_mode == "concat":
            condition_parts = []
            if self.include_wrist_query_in_action:
                condition_parts.append(adapter_out["q_wrist"])
            condition_parts.append(adapter_out["c_wrist"])
            if self.include_future_wrist_in_action:
                condition_parts.append(pred_future)
            condition_tokens = torch.cat(condition_parts, dim=1)
            last_hidden = torch.cat([hidden_for_condition, condition_tokens], dim=1)
            if action_context_mask is not None:
                condition_mask = torch.ones(
                    condition_tokens.shape[:2],
                    device=action_context_mask.device,
                    dtype=torch.bool,
                )
                action_context_mask = torch.cat(
                    [action_context_mask.to(torch.bool), condition_mask],
                    dim=1,
                )
        else:
            raise ValueError(f"Unknown M2W action_condition_mode: {self.action_condition_mode}")
        return last_hidden, state_tensor, wrist_latent_loss, action_context_mask

    def get_action_condition(
        self,
        batch_images,
        wrist_views,
        future_wrist_views,
        instructions,
        cot_targets=None,
        state=None,
        future_loss_weights=None,
        compute_cot_loss: bool = True,
    ):
        if self.cot_reasoning_mode in {
            "latent_query",
            "parallel_query",
            "implicit_query",
            "qwen_implicit",
            "implicit_cot",
        }:
            hidden, h_reason, cot_loss, action_context_mask = self._qwen_forward_implicit_query_cot(
                batch_images=batch_images,
                instructions=instructions,
                cot_targets=cot_targets,
                compute_cot_loss=compute_cot_loss,
            )
            last_hidden, state_tensor, wrist_latent_loss, action_context_mask = (
                self._condition_from_qwen_hidden(
                    hidden=hidden,
                    h_reason=h_reason,
                    wrist_views=wrist_views,
                    future_wrist_views=future_wrist_views,
                    state=state,
                    future_loss_weights=future_loss_weights,
                    action_context_mask=action_context_mask,
                )
            )
            return last_hidden, state_tensor, cot_loss, wrist_latent_loss, action_context_mask

        # A true no-CoT baseline must remove the assistant solution from the
        # Qwen sequence, not merely multiply its CE loss by zero. Otherwise the
        # DiT can still condition on teacher-forced CoT hidden states.
        if not self.use_cot_loss:
            cot_targets = None
            compute_cot_loss = False
        hidden, h_reason, cot_loss, action_context_mask = self._qwen_forward(
            batch_images,
            instructions,
            cot_targets,
            compute_cot_loss=compute_cot_loss,
        )
        last_hidden, state_tensor, wrist_latent_loss, action_context_mask = self._condition_from_qwen_hidden(
            hidden=hidden,
            h_reason=h_reason,
            wrist_views=wrist_views,
            future_wrist_views=future_wrist_views,
            state=state,
            future_loss_weights=future_loss_weights,
            action_context_mask=action_context_mask,
        )
        return last_hidden, state_tensor, cot_loss, wrist_latent_loss, action_context_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        (
            batch_images,
            wrist_views,
            future_wrist_views,
            instructions,
            cot_targets,
            state,
            future_loss_weights,
        ) = self.align_model_input(examples)

        last_hidden, state, cot_loss, wrist_latent_loss, action_context_mask = self.get_action_condition(
            batch_images=batch_images,
            wrist_views=wrist_views,
            future_wrist_views=future_wrist_views,
            instructions=instructions,
            cot_targets=cot_targets,
            state=state,
            future_loss_weights=future_loss_weights,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array([example["action"] for example in examples]),
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated = self.config.trainer.get(
                "repeated_diffusion_steps",
                self.config.framework.action_model.get("repeated_diffusion_steps", 8),
            )
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

        total_loss = action_loss + self.lambda_cot * cot_loss + self.lambda_wrist * wrist_latent_loss
        return {
            "action_loss": action_loss,
            "cot_loss": cot_loss,
            "wrist_latent_loss": wrist_latent_loss,
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        (
            batch_images,
            wrist_views,
            future_wrist_views,
            instructions,
            cot_targets,
            state,
            future_loss_weights,
        ) = self.align_model_input(examples)

        generated_cot = None
        if self.generate_cot_at_inference:
            hidden, h_reason, generated_cot, action_context_mask = self._generate_cot_with_hidden(
                batch_images,
                instructions,
            )
            if self.inference_empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            last_hidden, state, _, action_context_mask = self._condition_from_qwen_hidden(
                hidden=hidden,
                h_reason=h_reason,
                wrist_views=wrist_views,
                future_wrist_views=None,
                state=state,
                future_loss_weights=None,
                action_context_mask=action_context_mask,
            )
        else:
            last_hidden, state, _, _, action_context_mask = self.get_action_condition(
                batch_images=batch_images,
                wrist_views=wrist_views,
                future_wrist_views=None,
                instructions=instructions,
                cot_targets=None,
                state=state,
                future_loss_weights=None,
                compute_cot_loss=False,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                encoder_attention_mask=action_context_mask,
            )

        result = {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
        if generated_cot is not None:
            result["generated_cot"] = generated_cot
        return result
