# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Adapted from upstream StarVLA.

import argparse
import logging
import os
import socket

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _load_w2_aware_from_pretrained(ckpt_path: str):
    """Load a W2 checkpoint using the framework recorded in config.yaml."""
    from starVLA.model.framework import build_framework
    from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config

    model_config, norm_stats = read_mode_config(ckpt_path)
    framework_cfg = model_config.get("framework", {})
    framework_name = str(
        framework_cfg.get("name") or framework_cfg.get("framework_py") or ""
    )
    is_w2_m2w = framework_name.startswith("QwenSubtaskM2W") and "w2" in framework_name
    if not is_w2_m2w:
        framework_name = "QwenSubtaskM2W_w2"
        logging.info(
            "No W2 framework name found in checkpoint config; using %s.",
            framework_name,
        )
    else:
        logging.info("Loading W2 policy checkpoint with %s.", framework_name)
    framework_cfg["name"] = framework_name

    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    model = build_framework(cfg=config)
    model.norm_stats = norm_stats

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model_keys = set(model.state_dict())
        checkpoint_keys = set(state)
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        if missing_keys:
            logging.warning("Missing keys in state_dict: %s", missing_keys)
        if unexpected_keys:
            logging.warning("Unexpected keys in state_dict: %s", unexpected_keys)
        raise
    return model


def _keep_m2w_world_model_fp32(vla) -> None:
    modules = []
    visual_encoder = getattr(vla, "visual_encoder", None)
    if visual_encoder is not None:
        modules.append(("visual_encoder", visual_encoder, True))
    jepa_predictor = getattr(vla, "jepa_predictor", None)
    if jepa_predictor is not None:
        modules.append(("jepa_predictor", jepa_predictor, False))
    wrist_context_adapter = getattr(vla, "wrist_context_adapter", None)
    if wrist_context_adapter is not None:
        modules.append(("wrist_context_adapter", wrist_context_adapter, False))
    if not modules:
        return

    logging.info(
        "Keeping M2W world-model modules in fp32 during bf16 policy serving: %s",
        ", ".join(name for name, _, _ in modules),
    )
    for _, module, freeze in modules:
        module.float()
        module.eval()
        if freeze:
            for param in module.parameters():
                param.requires_grad = False


def main(args) -> None:
    vla = _load_w2_aware_from_pretrained(args.ckpt_path)
    if args.disable_cot_generation and hasattr(vla, "generate_cot_at_inference"):
        logging.info("Disabling generated CoT at inference for policy server.")
        vla.generate_cot_at_inference = False
    if (
        args.inference_cot_max_new_tokens is not None
        and hasattr(vla, "inference_cot_max_new_tokens")
    ):
        logging.info(
            "Setting inference_cot_max_new_tokens=%d",
            args.inference_cot_max_new_tokens,
        )
        vla.inference_cot_max_new_tokens = args.inference_cot_max_new_tokens
    if args.inference_cot_use_cache and hasattr(vla, "inference_cot_use_cache"):
        logging.info("Setting inference_cot_use_cache=True")
        vla.inference_cot_use_cache = True
    if args.inference_empty_cache and hasattr(vla, "inference_empty_cache"):
        logging.info("Setting inference_empty_cache=True")
        vla.inference_empty_cache = True

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
        _keep_m2w_world_model_fp32(vla)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--idle_timeout",
        type=int,
        default=1800,
        help="Idle timeout in seconds; -1 means never close.",
    )
    parser.add_argument(
        "--disable_cot_generation",
        action="store_true",
        help="Skip explicit CoT generation during evaluation.",
    )
    parser.add_argument(
        "--inference_cot_max_new_tokens",
        type=int,
        default=None,
        help="Override the maximum number of generated CoT tokens.",
    )
    parser.add_argument(
        "--inference_cot_use_cache",
        action="store_true",
        help="Use the Qwen KV cache for CoT generation.",
    )
    parser.add_argument(
        "--inference_empty_cache",
        action="store_true",
        help="Call torch.cuda.empty_cache after generated CoT.",
    )
    return parser


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
