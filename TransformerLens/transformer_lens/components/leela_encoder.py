import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from transformer_lens.hook_points import HookPoint
from transformer_lens.components import (
    LayerNorm,
)

class SmolGen(nn.Module):
    """SmolGen module"""
    
    def __init__(self, d_model: int = 768, n_heads: int = 24, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.compress = nn.Linear(d_model, 32, bias=False)
        self.dense1 = nn.Linear(2048, 256)
        self.ln1 = nn.LayerNorm(256, eps=eps)
        self.dense2 = nn.Linear(256, 256 * n_heads)
        self.ln2 = nn.LayerNorm(256 * n_heads, eps=eps)
        self.smol_weight_gen = nn.Linear(256, 4096, bias=False)
        self.hook_output = HookPoint()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 64, 768]
        Returns:
            weights: [batch_size, 64, 64]
        """
        batch_size, seq_len, _ = x.shape
        
        # compress
        compressed = self.compress(x)
        
        # reshape for dense1
        x_flat = compressed.view(batch_size, -1)
        
        # dense1 + swish + ln1
        x = self.dense1(x_flat)
        x = F.silu(x)
        x = self.ln1(x)
        
        # dense2 + swish + ln2
        x = self.dense2(x)
        x = F.silu(x)
        x = self.ln2(x)
        
        # reshape for smol_weight_gen
        x = x.view(batch_size, self.n_heads, 256)

        weights = self.smol_weight_gen(x)
        weights = self.hook_output(weights.view(batch_size, self.n_heads, 64, 64))
        return weights

class LC0MLP(nn.Module):
    """MLP for LC0"""
    
    def __init__(self, d_model, d_ff, act_fn="squaredrelu"):
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_ff)
        self.dense2 = nn.Linear(d_ff, d_model)
        self.act_fn = act_fn
        
    def forward(self, x):
        x = x
        x = self.dense1(x)
        
        if self.act_fn == "squaredrelu":
            x = torch.nn.functional.relu(x) ** 2
        elif self.act_fn == "mish":
            x = F.mish(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.act_fn}")
            
        x = self.dense2(x)
        return x

class Lc0LayerNorm(nn.Module):
    """LayerNorm implementation identical to nn.LayerNorm, with hook support added."""
    
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        # Add hook_scale for gradient control
        self.hook_ln_input = HookPoint()
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        # Hook_normalized is on the LN output
        self.hook_normalized = HookPoint()  # [batch, pos, d_model]

    def forward(self, x):
        x = self.hook_ln_input(x)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        
        # Compute scale and pass through hook
        scale = torch.sqrt(var + self.eps)
        scale = self.hook_scale(scale)  # Add hook support
        
        # Normalize
        x = (x - mean) / scale
        
        # Apply weights and bias, and pass through hook_normalized
        return self.hook_normalized(x * self.w + self.b)

class MultiHeadAttention(nn.Module):
    """Multi-head attention - integrates SmolGen logic."""
    
    def __init__(self, d_model: int = 1024, n_heads: int = 32, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.qk_scale = nn.Parameter(torch.tensor([1.0 / (self.d_k ** 0.5)]))

        # Hook points
        self.hook_attn_score = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_pattern = HookPoint()  # Attention pattern hook
        self.hook_z = HookPoint()  # Attention output hook
        
        # Integrate SmolGen
        self.smolgen = SmolGen(d_model, n_heads, eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            attn_out: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape   
        
        # Q, K, V projections
        q = self.hook_q(self.q_proj(x))
        k = self.hook_k(self.k_proj(x))
        v = self.hook_v(self.v_proj(x))
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = self.hook_attn_score(torch.matmul(q, k.transpose(-2, -1)) * self.qk_scale)
        
        # SmolGen generates weights
        smol_weights = self.smolgen(x)
        
        # Apply SmolGen weights to attention scores
        combined_scores = scores + smol_weights
        attn_weights = self.hook_attn_pattern(F.softmax(combined_scores, dim=-1)) #[B, H, S, S]
        
        # Compute attention output
        attn_out = torch.matmul(attn_weights, v) #[B, H, S, head_dim]
        attn_out = self.hook_z(attn_out)
        
        # Reshape back to original shape
        attn_out = (
            attn_out
            .permute(0, 2, 1, 3)  # [B, S, H, head_dim]
            .contiguous()
            .view(batch_size, seq_len, self.d_model)  # [B, S, D]
        )
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        
        return attn_out

class EncoderLayer(nn.Module):
    """Encoder layer."""
    
    def __init__(self, d_model: int = 768, n_heads: int = 24, d_ff: int = 1024, 
                 mlp_act_fn: str = "squaredrelu", resid_alpha: str = "pre", eps: float = 1e-5):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.resid_alpha = resid_alpha
        
        # Multi-head attention (now includes SmolGen logic)
        self.mha = MultiHeadAttention(d_model, n_heads, eps)
        
        # Layer normalization
        self.ln1 = Lc0LayerNorm(d_model, eps)
        self.ln2 = Lc0LayerNorm(d_model, eps)

        # Hook points
        self.hook_attn_in = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.resid_mid_after_ln = HookPoint()
        self.resid_post_after_ln = HookPoint()
        
        # FFN layers
        self.mlp = LC0MLP(d_model, d_ff, mlp_act_fn)
        
        # Alpha parameters for residual connections
        self.alpha_input = nn.Parameter(torch.ones(1))
        self.alpha_out1 = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for residual connection
        residual = self.hook_attn_in(x)
        
        # Multi-head attention (now directly returns complete attn_out)
        attn_out = self.mha(x)
        attn_out = self.hook_attn_out(attn_out)
    
        # First residual connection - according to resid_alpha configuration
        if self.resid_alpha == "pre":
            x = attn_out + (residual * self.alpha_input)
        elif self.resid_alpha == "post":
            x = (attn_out * self.alpha_input) + residual
        else:
            raise ValueError(f"Unsupported resid_alpha: {self.resid_alpha}")

        x = self.resid_mid_after_ln(self.ln1(x))
        
        # Save for second residual connection
        residual2 = x
        
        # mlp
        mlp_out = self.hook_mlp_out(self.mlp(x))
        
        # Second residual connection - according to resid_alpha configuration
        if self.resid_alpha == "pre":
            x = mlp_out + (residual2 * self.alpha_out1)
        elif self.resid_alpha == "post":
            x = (mlp_out * self.alpha_out1) + residual2

        x = self.resid_post_after_ln(self.ln2(x))
        
        return x