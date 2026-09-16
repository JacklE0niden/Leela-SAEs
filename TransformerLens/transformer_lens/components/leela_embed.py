import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens.hook_points import HookPoint


class LeelaEmbed(nn.Module):
    """Leela Embed"""
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embedding = nn.Linear(176, d_model)
        
        # MA gating机制
        self.ma_gating_mul = nn.Parameter(torch.randn(64, d_model))
        self.ma_gating_add = nn.Parameter(torch.randn(64, d_model))
        
        # 位置编码参数
        self.pos_encoding_base = nn.Parameter(torch.randn(1, 64, 64))
        self.hook_after_position_embedding = HookPoint()
        self.hook_input_embedding = HookPoint() 
        self.hook_ma_gating = HookPoint()
        self.hook_input = HookPoint()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 112, 8, 8] - 输入特征平面
        Returns:
            output: [batch_size, 64, d_model] - 嵌入后的特征
        """
        x = self.hook_input(x)
        batch_size = x.shape[0]
        
        # 转置和重塑: [batch_size, 112, 8, 8] -> [batch_size, 64, 112]
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, 64, 112)
    
        # 位置编码
        pos_encoding = self.pos_encoding_base.expand(batch_size, 64, 64)
        
        # 拼接输入特征和位置编码
        x = self.hook_after_position_embedding(torch.cat([x, pos_encoding], dim=-1))
        
        # 输入嵌入
        x = self.hook_input_embedding(self.input_embedding(x))
        # Mish激活
        x = F.mish(x)
        # MA gating
        x = x * self.ma_gating_mul.unsqueeze(0)    
        x = self.hook_ma_gating(x + self.ma_gating_add.unsqueeze(0))
        return x
    
    

class BT4LeelaEmbed(nn.Module):
    def __init__(self, d_model: int = 1024):  # 根据实际输出改为1024
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
        """
        Args:
            x: [batch_size, 112, 8, 8] - 输入特征平面
        Returns:
            output: [batch_size, 64, d_model] - 嵌入后的特征
        """
        x = self.hook_input(x)
        batch_size = x.shape[0]

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, 64, 112)

        pos_slice = x[:, :, :12]
        pos_reshaped = pos_slice.reshape(batch_size, -1)  # [B, 768]
        
        pos_processed = self.embedding_preprocess(pos_reshaped)  # [B, 32768]
        pos_processed = pos_processed.reshape(batch_size, 64, 512)  # [B, 64, 512]
        x_concat = torch.cat([x, pos_processed], dim=-1)  # [B, 64, 624]
        x_concat = x_concat.reshape(-1, 624)  # [B*64, 624]

        x = self.main_linear(x_concat)  # [B*64, 1024]

        x = F.mish(x)
        x = self.ln(x)  # [B*64, 1024]
        x = x.reshape(batch_size, 64, self.d_model)  # [B, 64, 1024]

        x_gated = x * self.ma_gating_mul.unsqueeze(0)  # ip_mul_gate
        x_gated = x_gated + self.ma_gating_add.unsqueeze(0)  # ip_add_gate
        
        x_gated = x_gated.reshape(-1, self.d_model)  # [B*64, 1024]

        residual = x_gated  # 保存残差连接的输入 [B*64, 1024]
        
        ffn_out = self.ffn_dense1(x_gated)  # [B*64, 1536]

        ffn_out = F.mish(ffn_out)  # mish激活
        ffn_out = self.ffn_dense2(ffn_out)  # [B*64, 1024]
        x = ffn_out * self.ffn_alpha + residual  # [B*64, 1024]
        x = self.ln2(x)  # [B*64, 1024]
        x = x.reshape(batch_size, 64, self.d_model)  # [B, 64, 1024]
        
        return x