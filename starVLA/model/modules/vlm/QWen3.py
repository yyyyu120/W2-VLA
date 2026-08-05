# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Adapted from upstream StarVLA.

import torch
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 153716 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = str(qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct"))

        import os
        from transformers import AutoConfig
        if model_id.startswith("."):
            model_id = os.path.abspath(model_id)

        if os.path.isdir(model_id):
            # Load config first to bypass huggingface_hub repo id validation bug
            model_config = AutoConfig.from_pretrained(model_id, local_files_only=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                config=model_config,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """
        label_mode = kwargs.get("label_mode", "action")
        return_solution_mask = kwargs.get("return_solution_mask", False)
        use_cot_prompt = kwargs.get("use_cot_prompt", True)
        supervise_assistant_eos = bool(kwargs.get("supervise_assistant_eos", True))

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if use_cot_prompt and "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=solutions is None,
            return_dict=True,
            return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            labels[:] = IGNORE_INDEX

            if label_mode == "assistant":
                solution_masks = torch.zeros_like(batch_inputs["input_ids"], dtype=torch.bool)
                tokenizer = self.processor.tokenizer
                assistant_end_ids = {
                    token_id
                    for token_id in (
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    )
                    if token_id is not None
                    and int(token_id) >= 0
                    and int(token_id) != getattr(tokenizer, "unk_token_id", None)
                }
                for i, solution in enumerate(solutions):
                    seq = batch_inputs["input_ids"][i]
                    solution_ids = tokenizer(
                        solution,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).input_ids.squeeze(0).to(seq.device)

                    if solution_ids.numel() == 0 or solution_ids.numel() > seq.numel():
                        continue

                    start_idx = None
                    for pos in range(seq.numel() - solution_ids.numel(), -1, -1):
                        if torch.equal(seq[pos : pos + solution_ids.numel()], solution_ids):
                            start_idx = pos
                            break

                    if start_idx is not None:
                        end_idx = start_idx + solution_ids.numel()
                        labels[i, start_idx:end_idx] = seq[start_idx:end_idx]
                        solution_masks[i, start_idx:end_idx] = True
                        if (
                            supervise_assistant_eos
                            and end_idx < seq.numel()
                            and int(seq[end_idx].item()) in assistant_end_ids
                        ):
                            # Train explicit CoT spans to terminate.  Keep this
                            # out of solution_masks so h_reason/action context
                            # still pool only the actual CoT content tokens.
                            labels[i, end_idx] = seq[end_idx]
                    else:
                        RuntimeWarning(
                            "assistant solution tokens were not found in the tokenized sequence; "
                            "masking this sample out of COT loss."
                        )

                if return_solution_mask:
                    batch_inputs["solution_mask"] = solution_masks
            else:
                action_token_min = _ACTION_TOKEN_MIN
                action_token_max = _ACTION_TOKEN_MAX
                labels = batch_inputs["input_ids"].clone()
                for i in range(labels.size(0)):
                    seq = labels[i]
                    mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                    nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                    if nonzero_indices.numel() > 0:
                        first_action_index = nonzero_indices[0].item()
                        seq[:first_action_index] = IGNORE_INDEX
                    else:
                        seq[:] = IGNORE_INDEX
                        RuntimeWarning(
                            "action tokens are not in your tokenizer; "
                            "see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                        )

            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)




if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
