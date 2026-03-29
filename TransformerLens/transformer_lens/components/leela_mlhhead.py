import torch
import torch.nn as nn
import torch.nn.functional as F



class MLHHead(nn.Module):
    """MLH头"""
    
    def __init__(self, d_model: int = 768, d_mlh_head: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.d_mlh_head = d_mlh_head
        self.embed = nn.Linear(d_model, d_mlh_head)
        self.dense1 = nn.Linear(d_mlh_head * 64, 128)
        self.dense2 = nn.Linear(128, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 64, d_model]
        Returns:
            mlh: [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # 嵌入层
        x = x.view(batch_size * 64, self.d_model)
        x = self.embed(x)
        x = F.mish(x)
        
        # 重塑并通过密集层
        x = x.view(batch_size, -1)
        x = self.dense1(x)
        x = F.mish(x)
        
        # 输出层
        mlh = self.dense2(x)
        mlh = F.mish(mlh)
        
        return mlh

