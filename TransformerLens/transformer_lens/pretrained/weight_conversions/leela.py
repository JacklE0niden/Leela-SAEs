# import torch
import numpy as np
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

import einops

def convert_leela_weights(load_dict, cfg: HookedTransformerConfig):
    """
    将LC0象棋模型的权重转换为TransformerLens格式
    
    Args:
        load_dict: LC0模型的原始权重字典
        cfg: TransformerLens配置对象
        
    Returns:
        state_dict: 转换后的权重字典
    """
    state_dict = {}
    
    # 1. 转换 attention_body -> embed（保留原参数名）
    for key, value in load_dict.items():
        if key.startswith('attention_body.'):
            # 提取相对路径（去掉attention_body.前缀）
            relative_path = key[len('attention_body.'):]
            # 使用embed.前缀
            new_key = f'embed.{relative_path}'
            state_dict[new_key] = value
    
    # 2. 转换 encoders -> blocks（保留所有原参数名）
    for key, value in load_dict.items():
        if key.startswith('encoders.'):
            # 将 encoders.X. 替换为 blocks.X.
            new_key = key.replace('encoders.', 'blocks.', 1)
            state_dict[new_key] = value
    # for key, value in load_dict.items():
    #     if key.startswith('encoders.'):
    #         new_key = key.replace('encoders.', 'blocks.', 1)
    #         # 只对blocks.0.mha.hook_q转置
    #         if new_key.endswith('q_proj.weight'):
    #             state_dict[new_key] = value.T
    #         else:
    #             state_dict[new_key] = value
    # 3. 保持其他头部不变 (policy_head, value_head, mlh_head)
    for key, value in load_dict.items():
        if key.startswith(('policy_head.', 'value_head.', 'mlh_head.')):
            state_dict[key] = value
    
    return state_dict
