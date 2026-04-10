# import torch
import numpy as np
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

import einops

def convert_leela_weights(load_dict, cfg: HookedTransformerConfig):
    """
    Convert LC0 chess model weights to TransformerLens format
    
    Args:
        load_dict: raw weight dictionary from the LC0 model
        cfg: TransformerLens config object
        
    Returns:
        state_dict: converted weight dictionary
    """
    state_dict = {}
    
    # 1. Convert attention_body -> embed (keep original parameter names)
    for key, value in load_dict.items():
        if key.startswith('attention_body.'):
            # Extract the relative path (remove the attention_body. prefix)
            relative_path = key[len('attention_body.'):]
            # Use the embed. prefix
            new_key = f'embed.{relative_path}'
            state_dict[new_key] = value
    
    # 2. Convert encoders -> blocks (keep all original parameter names)
    for key, value in load_dict.items():
        if key.startswith('encoders.'):
            # Replace encoders.X. with blocks.X.
            new_key = key.replace('encoders.', 'blocks.', 1)
            state_dict[new_key] = value
    # for key, value in load_dict.items():
    #     if key.startswith('encoders.'):
    #         new_key = key.replace('encoders.', 'blocks.', 1)
    #         # Only transpose blocks.0.mha.hook_q
    #         if new_key.endswith('q_proj.weight'):
    #             state_dict[new_key] = value.T
    #         else:
    #             state_dict[new_key] = value
    # 3. Keep other heads unchanged (policy_head, value_head, mlh_head)
    for key, value in load_dict.items():
        if key.startswith(('policy_head.', 'value_head.', 'mlh_head.')):
            state_dict[key] = value
    
    return state_dict
