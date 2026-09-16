import torch
import torch.nn as nn
from typing import Union, Optional

class ShiftRight(nn.Module):
    """
    将输入序列向右移动一位，在时间轴上添加padding。
    用于实现类似语言模型中的shift操作。
    """
    
    def __init__(self, pad_value: int = 0):
        """
        初始化ShiftRight模块。
        
        Args:
            pad_value: 用于填充的值，默认为0
        """
        super().__init__()
        self.pad_value = pad_value
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        对输入序列执行右移操作。
        
        Args:
            sequences: 输入张量，形状为 [batch_size, seq_len] 或 [batch_size, seq_len, feature_dim]
            
        Returns:
            右移后的张量，形状与输入相同
        """
        # print("sequences.shape before shift_right:", sequences.shape)
        
        # 获取batch_size
        batch_size = sequences.shape[0]
        
        # 创建BOS（beginning of sequence）数组
        if sequences.dim() == 2:
            # 2D情况: [batch_size, seq_len]
            bos_array = torch.full((batch_size, 1), self.pad_value, 
                                 dtype=sequences.dtype, device=sequences.device)
        else:
            # 3D情况: [batch_size, seq_len, feature_dim]
            feature_dim = sequences.shape[2]
            bos_array = torch.full((batch_size, 1, feature_dim), self.pad_value,
                                 dtype=sequences.dtype, device=sequences.device)
        
        # 在序列开头添加BOS token
        padded_sequences = torch.cat([bos_array, sequences], dim=1)
        
        # 移除最后一个位置，保持原始长度
        result = padded_sequences[:, :-1]
        
        # print("padded_sequences.shape after shift_right:", result.shape)
        return result
    
    def extra_repr(self) -> str:
        """返回模块的额外字符串表示。"""
        return f'pad_value={self.pad_value}'


# 为了保持向后兼容，也可以提供函数版本
def shift_right(sequences: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """
    将输入序列向右移动一位的函数版本。
    
    Args:
        sequences: 输入张量
        pad_value: 填充值，默认为0
        
    Returns:
        右移后的张量
    """
    shift_module = ShiftRight(pad_value=pad_value)
    return shift_module(sequences)