from typing import Dict, Union, List
import torch
import torch.nn as nn
import math
from typing import Optional

from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class SearchlessChessEmbed(nn.Module):
    """Searchless Chess embedding layer for HookedTransformer"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        max_sequence_length: int = 78,
        pos_encodings: str = "learned",  # "sinusoid" or "learned"
        emb_init_scale: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_sequence_length = max_sequence_length
        self.pos_encodings = pos_encodings
        self.emb_init_scale = emb_init_scale
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        if pos_encodings == "learned":
            self.pos_embed = nn.Embedding(max_sequence_length, embed_dim)
        else:
            self.pos_embed = None
            
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal distribution"""
        # Token embeddings initialization
        nn.init.trunc_normal_(self.token_embed.weight, std=self.emb_init_scale)
        
        # Position embeddings initialization (if learned)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed.weight, std=self.emb_init_scale)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Input token indices, shape [batch_size, seq_len]
        Returns:
            embeddings: Combined token and position embeddings, shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        
        embeddings = self.token_embed(tokens)
        embeddings = embeddings * math.sqrt(self.embed_dim)
        
        # Position embeddings
        if self.pos_encodings == "learned":
            positions = torch.arange(seq_len, device=tokens.device)
            pos_embeddings = self.pos_embed(positions)
            embeddings = embeddings + pos_embeddings
        else:  # sinusoid
            pos_embeddings = self._sinusoid_position_encoding(seq_len, self.embed_dim, tokens.device)
            embeddings = embeddings + pos_embeddings
        
        return embeddings
    
    def _sinusoid_position_encoding(
        self, 
        sequence_length: int, 
        hidden_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Generate sinusoid position encodings"""
        position = torch.arange(sequence_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, device=device) * 
                           -(math.log(10000.0) / hidden_size))
        
        pos_encoding = torch.zeros(sequence_length, hidden_size, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding

def embed_sequences_torch(
    sequences: torch.Tensor,
    vocab_size: int,
    embed_dim: int,
    max_sequence_length: int = 79,
    pos_encodings: str = "learned",
    emb_init_scale: float = 0.02,
) -> torch.Tensor:
    """
    PyTorch version of embed_sequences function
    
    Args:
        sequences: Input token sequences, shape [batch_size, seq_len]
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        max_sequence_length: Maximum sequence length for position embeddings
        pos_encodings: Type of position encoding ("sinusoid" or "learned")
        emb_init_scale: Initialization scale for embeddings
    
    Returns:
        embeddings: Combined token and position embeddings
    """
    embed_layer = SearchlessChessEmbed(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_sequence_length=max_sequence_length,
        pos_encodings=pos_encodings,
        emb_init_scale=emb_init_scale,
    )
    
    return embed_layer(sequences)
