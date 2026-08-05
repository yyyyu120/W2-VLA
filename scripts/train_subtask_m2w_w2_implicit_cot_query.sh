#!/usr/bin/env bash
set -euo pipefail

# W2 implicit CoT query:
#   - Qwen receives the compact CoT prompt plus latent query tokens.
#   - Qwen's LM head is supervised on GT CoT text during training.
#   - JEPA future-wrist prediction remains enabled; do not set skip_vjepa_init.
#   - Evaluation uses query hidden states directly and does not decode CoT text.
export cot_reasoning_mode=${cot_reasoning_mode:-implicit_query}
export latent_cot_num_queries=${latent_cot_num_queries:-16}
export qwen_action_context_mode=${qwen_action_context_mode:-exclude_cot_prompt}
export generate_cot_at_inference=${generate_cot_at_inference:-false}
export vlm_include_wrist_view=${vlm_include_wrist_view:-true}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_subtask_m2w_deepspeed.sh" "$@"
