import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.hook_points import HookPoint


class Mish(nn.Module):

    def __init__(self):
        super().__init__()
        self.hook_weight = HookPoint()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.tanh(F.softplus(x))
        weight = self.hook_weight(weight)
        return x * weight


class PolicyHead(nn.Module):
    def __init__(self, d_model: int = 768, policy_dim: int = 1858):
        super().__init__()
        
        self.hook_policy_input = HookPoint()
        self.dense1 = nn.Linear(d_model, d_model)
        self.mish = Mish()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = nn.Parameter(torch.ones(1))
        self.promotion = nn.Linear(d_model, 4, bias=False)
        self.promotion_weight = nn.Parameter(torch.randn(768, 4))
        self.indices = nn.Parameter(torch.randn(policy_dim))
        
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_policy_qk_score = HookPoint()
        self.hook_policy_out = HookPoint()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_policy_input(x)
        batch_size = x.shape[0]
        
        x = self.dense1(x)
        x = self.mish(x)
        
        q = self.hook_q(self.q_proj(x))
        k = self.hook_k(self.k_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = self.hook_policy_qk_score(scores * self.scale)
        
        promotion_slice = k[:, 56:64, :]
        promotion_out = self.promotion(promotion_slice)
        
        promotion_out = promotion_out.transpose(1, 2)

        promotion_out_part1, promotion_out_part2 = torch.split(promotion_out, [3, 1], dim=1)
        
        promotion_out = torch.add(promotion_out_part1, promotion_out_part2)

        promotion_out = promotion_out.transpose(1, 2)
        
        
        promotion_out = promotion_out.reshape(x.shape[0], 1, 24)
        
        promotion_slice2 = scores[:, 48:56, 56:64]
        promotion_out2 = promotion_slice2.reshape(-1, 64, 1)
        promotion_out2 = torch.cat([promotion_out2, promotion_out2, promotion_out2], dim=-1)
        
        promotion_out2 = promotion_out2.reshape(-1, 8, 24)
        
        promotion = promotion_out2 + promotion_out
        
        promotion = promotion.reshape(-1, 3, 64)
        
        policy = torch.cat([scores, promotion], dim=1)
    
        policy = policy.reshape(-1, 4288)
        
        indices_long = self.indices.detach().long()
        policy_logits = policy[:, indices_long]
        
        policy_logits = self.hook_policy_out(policy_logits)
        
        return policy_logits