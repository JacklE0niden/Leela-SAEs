import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from jaxtyping import Float


class SimplePolicyHead(nn.Module):
    """简单的线性层策略头，用于替代复杂的PolicyHead"""
    
    def __init__(self, d_model: int = 768, policy_dim: int = 1858, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.policy_dim = policy_dim
        
        # 简单的线性映射：从 [batch, 64, 768] 到 [batch, 1858]
        # 使用 global average pooling 将 64 个位置聚合到一个特征向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [batch, 768, 1]
        
        # 可选的中间层
        self.hidden_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(), 
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model // 4, policy_dim)
        
    def forward(self, x: Float[torch.Tensor, "batch seq d_model"]) -> Float[torch.Tensor, "batch policy_dim"]:
        """
        Args:
            x: [batch_size, 64, 768] - 从 transformer 输出的特征
        Returns:
            policy_logits: [batch_size, 1858] - 策略 logits
        """
        batch_size = x.shape[0]
        
        # 全局平均池化：[batch, 64, 768] -> [batch, 768, 64] -> [batch, 768, 1] -> [batch, 768]
        x = x.transpose(1, 2)  # [batch, 768, 64]
        x = self.global_pool(x)  # [batch, 768, 1]
        x = x.squeeze(-1)  # [batch, 768]
        
        # 通过隐藏层
        x = self.hidden_layer(x)  # [batch, 192]
        
        # 输出层
        policy_logits = self.output_layer(x)  # [batch, 1858]
        
        return policy_logits


class PositionAwarePolicyHead(nn.Module):
    """位置感知的Policy Head，保持棋盘位置信息"""
    
    def __init__(self, d_model: int = 768, policy_dim: int = 1858, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.policy_dim = policy_dim
        
        # 1. 位置编码 - 为64个棋盘位置添加位置信息
        self.position_embedding = nn.Parameter(torch.randn(64, d_model) * 0.02)
        
        # 2. 位置感知的特征提取
        self.position_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. 全局上下文聚合 - 使用注意力机制而不是简单平均
        self.global_context = nn.MultiheadAttention(
            embed_dim=d_model // 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. 位置特定的特征融合
        self.position_fusion = nn.Sequential(
            nn.Linear(d_model // 2 * 2, d_model // 2),  # 拼接局部和全局特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. 输出层
        self.output_layer = nn.Linear(d_model // 4, policy_dim)
        
        # 6. 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Float[torch.Tensor, "batch seq d_model"]) -> Float[torch.Tensor, "batch policy_dim"]:
        """
        Args:
            x: [batch_size, 64, 768] - 从 transformer 输出的特征
        Returns:
            policy_logits: [batch_size, 1858] - 策略 logits
        """
        batch_size, seq_len, d_model = x.shape
        assert seq_len == 64, f"Expected sequence length 64, got {seq_len}"
        
        # 1. 添加位置编码
        x = x + self.position_embedding.unsqueeze(0)  # [batch, 64, 768]
        
        # 2. 位置感知特征提取
        position_features = self.position_encoder(x)  # [batch, 64, 384]
        
        # 3. 全局上下文聚合 - 使用自注意力
        global_context, _ = self.global_context(
            position_features, position_features, position_features
        )  # [batch, 64, 384]
        
        # 4. 特征融合 - 拼接局部和全局特征
        fused_features = torch.cat([position_features, global_context], dim=-1)  # [batch, 64, 768]
        fused_features = self.position_fusion(fused_features)  # [batch, 64, 192]
        
        # 5. 全局聚合 - 使用加权平均而不是简单平均
        # 计算每个位置的权重
        position_weights = F.softmax(fused_features.mean(dim=-1, keepdim=True), dim=1)  # [batch, 64, 1]
        global_features = (fused_features * position_weights).sum(dim=1)  # [batch, 192]
        
        # 6. 输出层
        policy_logits = self.output_layer(global_features)  # [batch, 1858]
        
        return policy_logits


class EnhancedPositionAwarePolicyHead(nn.Module):
    """增强版位置感知Policy Head，使用更复杂的架构"""
    
    def __init__(self, d_model: int = 768, policy_dim: int = 1858, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.policy_dim = policy_dim
        
        # 1. 位置编码
        self.position_embedding = nn.Parameter(torch.randn(64, d_model) * 0.02)
        
        # 2. 多层位置编码器
        self.position_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model if i == 0 else d_model // 2, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(3)  # 3层编码器
        ])
        
        # 3. 棋盘结构感知 - 考虑棋盘的8x8结构
        self.board_attention = nn.MultiheadAttention(
            embed_dim=d_model // 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. 分层特征提取
        self.hierarchical_encoder = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 8),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 5. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model // 8, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, policy_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Float[torch.Tensor, "batch seq d_model"]) -> Float[torch.Tensor, "batch policy_dim"]:
        batch_size, seq_len, d_model = x.shape
        assert seq_len == 64, f"Expected sequence length 64, got {seq_len}"
        
        # 1. 添加位置编码
        x = x + self.position_embedding.unsqueeze(0)
        
        # 2. 多层位置编码
        for encoder in self.position_encoders:
            x = encoder(x)
        
        # 3. 棋盘结构注意力
        board_features, _ = self.board_attention(x, x, x)
        
        # 4. 分层特征提取
        hierarchical_features = self.hierarchical_encoder(board_features)
        
        # 5. 全局聚合 - 使用最大池化和平均池化的组合
        max_pooled = hierarchical_features.max(dim=1)[0]  # [batch, d_model//8]
        avg_pooled = hierarchical_features.mean(dim=1)    # [batch, d_model//8]
        global_features = max_pooled + avg_pooled         # [batch, d_model//8]
        
        # 6. 输出层
        policy_logits = self.output_layer(global_features)
        
        return policy_logits


class ChessSpecificPolicyHead(nn.Module):
    """专门为象棋设计的Policy Head，考虑象棋的特殊结构"""
    
    def __init__(self, d_model: int = 768, policy_dim: int = 1858, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.policy_dim = policy_dim
        
        # 1. 位置编码
        self.position_embedding = nn.Parameter(torch.randn(64, d_model) * 0.02)
        
        # 2. 棋盘区域编码 - 考虑不同区域的重要性
        # 中心区域 (d4, d5, e4, e5) 通常更重要
        self.center_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. 输出层
        self.output_layer = nn.Linear(d_model // 4, policy_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Float[torch.Tensor, "batch seq d_model"]) -> Float[torch.Tensor, "batch policy_dim"]:
        batch_size, seq_len, d_model = x.shape
        assert seq_len == 64, f"Expected sequence length 64, got {seq_len}"
        
        # 1. 添加位置编码
        x = x + self.position_embedding.unsqueeze(0)
        
        # 2. 中心区域注意力
        center_features, _ = self.center_attention(x, x, x)
        
        # 3. 特征提取
        features = self.feature_extractor(center_features)
        
        # 4. 全局聚合 - 使用加权平均，中心位置权重更高
        # 创建中心权重掩码
        center_mask = torch.zeros(64, device=x.device)
        center_squares = [27, 28, 35, 36]  # d4, d5, e4, e5
        center_mask[center_squares] = 2.0
        center_mask = center_mask.unsqueeze(0).unsqueeze(-1)  # [1, 64, 1]
        
        # 加权平均
        weights = F.softmax(center_mask, dim=1)
        global_features = (features * weights).sum(dim=1)
        
        # 5. 输出层
        policy_logits = self.output_layer(global_features)
        
        return policy_logits


class AttentionBasedPolicyHead(nn.Module):
    """基于注意力的Policy Head，使用查询-键-值机制"""
    
    def __init__(self, d_model: int = 768, policy_dim: int = 1858, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.policy_dim = policy_dim
        
        # 1. 位置编码
        self.position_embedding = nn.Parameter(torch.randn(64, d_model) * 0.02)
        
        # 2. 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=12,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # 4. Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 5. 全局池化查询
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 6. 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, policy_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Float[torch.Tensor, "batch seq d_model"]) -> Float[torch.Tensor, "batch policy_dim"]:
        batch_size, seq_len, d_model = x.shape
        assert seq_len == 64, f"Expected sequence length 64, got {seq_len}"
        
        # 1. 添加位置编码
        x = x + self.position_embedding.unsqueeze(0)
        
        # 2. 自注意力 + 残差连接
        normed_x = self.norm1(x)
        attn_out, _ = self.multihead_attn(normed_x, normed_x, normed_x)
        x = x + attn_out
        
        # 3. 前馈网络 + 残差连接
        normed_x = self.norm2(x)
        ff_out = self.feed_forward(normed_x)
        x = x + ff_out
        
        # 4. 全局查询注意力
        global_query = self.global_query.expand(batch_size, -1, -1)
        global_context, _ = self.multihead_attn(global_query, x, x)
        global_context = global_context.squeeze(1)  # [batch, d_model]
        
        # 5. 输出投影
        policy_logits = self.output_projection(global_context)
        
        return policy_logits



