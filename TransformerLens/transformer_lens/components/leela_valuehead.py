import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    def __init__(self, d_model: int = 768, d_value_head: int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.d_value_head = d_value_head
        self.embed = nn.Linear(d_model, d_value_head)
        self.dense1 = nn.Linear(d_value_head * 64, 128)
        self.dense2 = nn.Linear(128, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = x.view(batch_size * 64, self.d_model)
        x = self.embed(x)
        x = F.mish(x)
        
        x = x.view(batch_size, -1)
        x = self.dense1(x)
        x = F.mish(x)
        
        x = self.dense2(x)
        wdl = F.softmax(x, dim=-1)
        
        return wdl