#!/bin/bash

set -euo pipefail

# Implicit CoT query ablation:
#   train:  dual images + compact CoT prompt + 16 latent query tokens + GT CoT text
#           Qwen's own LM head predicts the GT CoT text with CE.
#   action: DiT cross-attends to image tokens, raw instruction tokens, and query
#           token hidden states only.  It does not see CoT prompt rules or
#           teacher-forced CoT answer hidden states.
#   eval:   no autoregressive CoT decoding; one Qwen forward produces query states.
export cot_reasoning_mode=${cot_reasoning_mode:-implicit_query}
export latent_cot_num_queries=${latent_cot_num_queries:-16}
export action_condition_mode=${action_condition_mode:-qwen_only}
export qwen_input_views=${qwen_input_views:-dual}
export qwen_action_context_mode=${qwen_action_context_mode:-exclude_cot_prompt}
export use_cot_loss=${use_cot_loss:-true}
export lambda_cot=${lambda_cot:-0.1}
export generate_cot_at_inference=${generate_cot_at_inference:-false}
export use_wrist_future_loss=${use_wrist_future_loss:-false}
export lambda_wrist=${lambda_wrist:-0.0}
export wrist_history_frames=${wrist_history_frames:-1}
export load_wrist_future_views=${load_wrist_future_views:-false}
export skip_vjepa_init=${skip_vjepa_init:-true}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_subtask_m2w_train.sh"
