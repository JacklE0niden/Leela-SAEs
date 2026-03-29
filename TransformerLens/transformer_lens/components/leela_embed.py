import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens.hook_points import HookPoint


class LeelaEmbed(nn.Module):
    """Leela Embed"""
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        self.input_embedding = nn.Linear(176, d_model)
        
        self.ma_gating_mul = nn.Parameter(torch.randn(64, d_model))
        self.ma_gating_add = nn.Parameter(torch.randn(64, d_model))
        
        self.pos_encoding_base = nn.Parameter(torch.randn(1, 64, 64))
        self.hook_after_position_embedding = HookPoint()
        self.hook_input_embedding = HookPoint() 
        self.hook_ma_gating = HookPoint()
        self.hook_input = HookPoint()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
 
        x = self.hook_input(x)
        batch_size = x.shape[0]
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, 64, 112)
    
        pos_encoding = self.pos_encoding_base.expand(batch_size, 64, 64)
        
        x = self.hook_after_position_embedding(torch.cat([x, pos_encoding], dim=-1))
        
        x = self.hook_input_embedding(self.input_embedding(x))
        x = F.mish(x)
        # MA gating
        x = x * self.ma_gating_mul.unsqueeze(0)    
        x = self.hook_ma_gating(x + self.ma_gating_add.unsqueeze(0))
        return x
    
    

class BT4LeelaEmbed(nn.Module):
    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.d_model = d_model
        
        self.embedding_preprocess = nn.Linear(768, 32768)
        self.main_linear = nn.Linear(624, d_model)
        
        self.ma_gating_mul = nn.Parameter(torch.randn(64, d_model))
        self.ma_gating_add = nn.Parameter(torch.randn(64, d_model))
        
        self.ffn_dense1 = nn.Linear(d_model, 1536)
        self.ffn_dense2 = nn.Linear(1536, d_model)
        self.ffn_alpha = nn.Parameter(torch.ones(1))

        self.ln = nn.LayerNorm(d_model, eps=1e-3)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-3)
        
        self.hook_input = HookPoint()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_input(x)
        batch_size = x.shape[0]

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, 64, 112)
        
        # Position embedding

        pos_slice = x[:, :, :12]
        pos_reshaped = pos_slice.reshape(batch_size, -1)
        
        pos_processed = self.embedding_preprocess(pos_reshaped)
        pos_processed = pos_processed.reshape(batch_size, 64, 512)
        x_concat = torch.cat([x, pos_processed], dim=-1)
        x_concat = x_concat.reshape(-1, 624)

        x = self.main_linear(x_concat)

        x = F.mish(x)
        x = self.ln(x)
        x = x.reshape(batch_size, 64, self.d_model)

        x_gated = x * self.ma_gating_mul.unsqueeze(0)
        x_gated = x_gated + self.ma_gating_add.unsqueeze(0)
        
        x_gated = x_gated.reshape(-1, self.d_model)

        residual = x_gated
        
        ffn_out = self.ffn_dense1(x_gated)

        ffn_out = F.mish(ffn_out)
        ffn_out = self.ffn_dense2(ffn_out)
        x = ffn_out * self.ffn_alpha + residual
        x = self.ln2(x)
        x = x.reshape(batch_size, 64, self.d_model)
        
        return x