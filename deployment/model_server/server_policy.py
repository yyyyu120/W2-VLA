# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Adapted from upstream StarVLA.

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )
    if args.disable_cot_generation and hasattr(vla, "generate_cot_at_inference"):
        logging.info("Disabling generated CoT at inference for policy server.")
        vla.generate_cot_at_inference = False
    if args.inference_cot_max_new_tokens is not None and hasattr(vla, "inference_cot_max_new_tokens"):
        logging.info("Setting inference_cot_max_new_tokens=%d", args.inference_cot_max_new_tokens)
        vla.inference_cot_max_new_tokens = args.inference_cot_max_new_tokens
    if args.inference_cot_use_cache and hasattr(vla, "inference_cot_use_cache"):
        logging.info("Setting inference_cot_use_cache=True")
        vla.inference_cot_use_cache = True
    if args.inference_empty_cache and hasattr(vla, "inference_empty_cache"):
        logging.info("Setting inference_empty_cache=True")
        vla.inference_empty_cache = True
    if args.num_inference_timesteps is not None:
        action_model = getattr(vla, "action_model", None)
        if action_model is None or not hasattr(action_model, "num_inference_timesteps"):
            logging.warning(
                "Ignoring num_inference_timesteps=%d because this action head "
                "does not expose flow-matching inference steps.",
                args.num_inference_timesteps,
            )
        else:
            old_steps = int(action_model.num_inference_timesteps)
            action_model.num_inference_timesteps = args.num_inference_timesteps
            if hasattr(action_model, "config"):
                action_model.config.num_inference_timesteps = args.num_inference_timesteps
            logging.info(
                "Overriding flow-matching inference steps: %d -> %d",
                old_steps,
                args.num_inference_timesteps,
            )

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    visual_encoder = getattr(vla, "visual_encoder", None)
    if args.keep_vjepa_fp32 and visual_encoder is not None:
        visual_encoder.float()
        logging.info("Keeping frozen V-JEPA encoder in FP32; main policy remains BF16.")
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
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--keep_vjepa_fp32",
        action="store_true",
        help="Keep the frozen V-JEPA encoder in FP32 after converting the main policy to BF16.",
    )
    parser.add_argument("--idle_timeout", type=int, default=1800,
                        help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--disable_cot_generation", action="store_true",
                        help="For QwenSubtaskM2W, skip explicit CoT generation during eval to reduce memory.")
    parser.add_argument("--inference_cot_max_new_tokens", type=int, default=None,
                        help="Override QwenSubtaskM2W inference CoT generation length.")
    parser.add_argument("--inference_cot_use_cache", action="store_true",
                        help="Use KV cache for QwenSubtaskM2W CoT generation during eval.")
    parser.add_argument("--inference_empty_cache", action="store_true",
                        help="Call torch.cuda.empty_cache after generated CoT during eval.")
    parser.add_argument("--num_inference_timesteps", type=int, default=None,
                        help="Override flow-matching action-head inference steps.")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
