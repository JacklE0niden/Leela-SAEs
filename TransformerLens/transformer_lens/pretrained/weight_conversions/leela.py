# import torch
import numpy as np
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

import einops

def convert_leela_weights(load_dict, cfg: HookedTransformerConfig):
    state_dict = {}
    for key, value in load_dict.items():
        if key.startswith('attention_body.'):
            relative_path = key[len('attention_body.'):]
            new_key = f'embed.{relative_path}'
            state_dict[new_key] = value
    
    for key, value in load_dict.items():
        if key.startswith('encoders.'):
            new_key = key.replace('encoders.', 'blocks.', 1)
            state_dict[new_key] = value
    for key, value in load_dict.items():
        if key.startswith(('policy_head.', 'value_head.', 'mlh_head.')):
            state_dict[key] = value
    
    return state_dict
