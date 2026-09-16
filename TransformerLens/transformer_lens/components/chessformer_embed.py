"""Hooked Transformer ChessFormer Embed Component.

This module contains all the component :class:`Embed`.
"""
from typing import Dict, Union, List

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

# 添加ChessFormer相关的常量
MAX_HALFMOVES = 128
MAX_FULLMOVES = 256
EMPTY_SQ_IDX = 12

# 棋子到索引的映射
PIECE_TO_IDX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # 白方棋子
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,  # 黑方棋子
}

# 棋盘位置到索引的映射
SQUARE_TO_IDX = {
    'a1': 0, 'b1': 1, 'c1': 2, 'd1': 3, 'e1': 4, 'f1': 5, 'g1': 6, 'h1': 7,
    'a2': 8, 'b2': 9, 'c2': 10, 'd2': 11, 'e2': 12, 'f2': 13, 'g2': 14, 'h2': 15,
    'a3': 16, 'b3': 17, 'c3': 18, 'd3': 19, 'e3': 20, 'f3': 21, 'g3': 22, 'h3': 23,
    'a4': 24, 'b4': 25, 'c4': 26, 'd4': 27, 'e4': 28, 'f4': 29, 'g4': 30, 'h4': 31,
    'a5': 32, 'b5': 33, 'c5': 34, 'd5': 35, 'e5': 36, 'f5': 37, 'g5': 38, 'h5': 39,
    'a6': 40, 'b6': 41, 'c6': 42, 'd6': 43, 'e6': 44, 'f6': 45, 'g6': 46, 'h6': 47,
    'a7': 48, 'b7': 49, 'c7': 50, 'd7': 51, 'e7': 52, 'f7': 53, 'g7': 54, 'h7': 55,
    'a8': 56, 'b8': 57, 'c8': 58, 'd8': 59, 'e8': 60, 'f8': 61, 'g8': 62, 'h8': 63,
}

# 索引到UCI移动的映射（简化版本）
IDX_TO_UCI_MOVE = {i: f"move_{i}" for i in range(1969)}

# Embed & Unembed
class ChessFormerEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        
        self.side_embed = nn.Embedding(2, self.cfg.d_model, dtype=self.cfg.dtype)
        self.castling_embed_k = nn.Parameter(torch.randn(1,1,self.cfg.d_model,dtype=self.cfg.dtype))
        self.castling_embed_q = nn.Parameter(torch.randn(1,1,self.cfg.d_model,dtype=self.cfg.dtype))
        self.castling_embed_K = nn.Parameter(data=torch.randn(1,1,self.cfg.d_model,dtype=self.cfg.dtype))
        self.castling_embed_Q = nn.Parameter(torch.randn(1,1,self.cfg.d_model,dtype=self.cfg.dtype))
        self.no_castling_embed = nn.Parameter(torch.randn(1,1,self.cfg.d_model,dtype=self.cfg.dtype))
        self.piece_embed = nn.Embedding(13,self.cfg.d_model,dtype=self.cfg.dtype)
        
        self.no_en_passant_embed = nn.Parameter(torch.randn(1,1,self.cfg.d_model,dtype=self.cfg.dtype))
        self.half_move_embed = nn.Embedding(128,self.cfg.d_model,dtype=self.cfg.dtype)
        self.full_move_embed = nn.Embedding(256,self.cfg.d_model,dtype=self.cfg.dtype)
        self.repetition_embed = nn.Embedding(3,self.cfg.d_model,dtype=self.cfg.dtype)
        self.pos_embed = nn.Embedding(64,self.cfg.d_model,dtype=self.cfg.dtype)
        
    def _parse_fen_string(self, fen_str: str) -> Dict:
        parts = fen_str.split()
        if len(parts) != 6:
            raise ValueError(f"Invalid FEN string: {fen_str}. Expected 6 fields")
        return {
            "piece_placement": parts[0],
            "side_to_move": parts[1],
            "castling": parts[2],
            "en_passant": parts[3],
            "halfmove_clock": parts[4],
            "fullmove_number": parts[5],
        }

    def forward(self, fen_list: List[str], repetitions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fen: List of fen strings
        
        Returns:
            torch tensor of shape (n_fen,73,hidden_size) where 73 tokens consists of:
                64 piece tokens (fen's first field) +
                1 which-side-to-move token (fen's second field) +
                4 casting rights tokens (fen's third field) + 
                1 en-passant target token (fen's fourth field) + 
                1 half move clock token (fen's fifth field) +
                1 full move number token (fen's fifth field) +
                1 repetition count token (repetitions input)
        """
        batch_size = len(fen_list)
        print("batch_size:", batch_size)
        print("repetitions:", repetitions)
        assert batch_size == repetitions.shape[0]
        assert len(repetitions.size()) == 1
        batch_tokens = []
        device = self.side_embed.weight.device

        # Precompute all square indices
        square_indices = torch.arange(64, device=device)
        all_pos_embeds = self.pos_embed(square_indices) # (64,D)

        for fen_str in fen_list:
            parsed_fen = self._parse_fen_string(fen_str)
            tokens = []

            # --- 1. Piece Placement (64 tokens) ---
            piece_indices = torch.full((64,), EMPTY_SQ_IDX, dtype=torch.long, device=device)
            current_rank = 7 # Start from rank 8
            current_file = 0 # Start from file 'a'
            for char in parsed_fen["piece_placement"]:
                if char == '/':
                    current_rank -= 1
                    current_file = 0
                elif char.isdigit():
                    current_file += int(char)
                elif char in PIECE_TO_IDX:
                    sq_idx = current_rank * 8 + current_file
                    if 0 <= sq_idx < 64:
                         piece_indices[sq_idx] = PIECE_TO_IDX[char]
                    else:
                         raise ValueError(f"Invalid FEN piece placement: {parsed_fen['piece_placement']}")
                    current_file += 1
                else:
                     raise ValueError(f"Invalid character in FEN piece placement: {char}")

            piece_embeds = self.piece_embed(piece_indices) # (64, D)
            # Add positional embeddings
            board_tokens = piece_embeds + all_pos_embeds # (64, D)
            tokens.append(board_tokens)

            # --- 2. Side to Move (1 token) ---
            side_idx = 0 if parsed_fen["side_to_move"] == 'w' else 1
            side_token = self.side_embed(torch.tensor(side_idx, device=device)).unsqueeze(0) # (1, D)
            tokens.append(side_token)

            # --- 3. Castling Rights (4 tokens) ---
            castling_str = parsed_fen["castling"]
            castling_tokens = torch.cat([
                self.castling_embed_K if 'K' in castling_str else self.no_castling_embed.expand(1, 1, -1),
                self.castling_embed_Q if 'Q' in castling_str else self.no_castling_embed.expand(1, 1, -1),
                self.castling_embed_k if 'k' in castling_str else self.no_castling_embed.expand(1, 1, -1),
                self.castling_embed_q if 'q' in castling_str else self.no_castling_embed.expand(1, 1, -1)
            ], dim=1).squeeze(0) # (4, D)
            tokens.append(castling_tokens)

            # --- 4. En Passant Target (1 token) ---
            en_passant_str = parsed_fen["en_passant"]
            if en_passant_str == '-':
                en_passant_token = self.no_en_passant_embed.squeeze(0) # (1, D)
            else:
                if en_passant_str in SQUARE_TO_IDX:
                    sq_idx = SQUARE_TO_IDX[en_passant_str]
                    en_passant_token = self.pos_embed(torch.tensor(sq_idx, device=device)).unsqueeze(0) # (1, D)
                else:
                    raise ValueError(f"Invalid en passant square: {en_passant_str}")
            tokens.append(en_passant_token)

            # --- 5. Half Move Clock (1 token) ---
            try:
                half_move_int = int(parsed_fen["halfmove_clock"])
            except ValueError:
                 raise ValueError(f"Invalid halfmove clock value: {parsed_fen['halfmove_clock']}")
            # Clamp value before embedding lookup
            half_move_clamped = torch.clamp(torch.tensor(half_move_int, device=device), 0, MAX_HALFMOVES - 1)
            half_move_token = self.half_move_embed(half_move_clamped).unsqueeze(0) # (1, D)
            tokens.append(half_move_token)

            # --- 6. Full Move Number (1 token) ---
            try:
                full_move_int = int(parsed_fen["fullmove_number"])
            except ValueError:
                 raise ValueError(f"Invalid fullmove number value: {parsed_fen['fullmove_number']}")
             # Clamp value (min 1 for full moves) before embedding lookup (adjusting for 0-based index)
            full_move_clamped = torch.clamp(torch.tensor(full_move_int, device=device), 1, MAX_FULLMOVES) - 1
            full_move_token = self.full_move_embed(full_move_clamped).unsqueeze(0) # (1, D)
            tokens.append(full_move_token)

            # Concatenate all tokens for this FEN string
            # Shapes: (64, D), (1, D), (4, D), (1, D), (1, D), (1, D) -> Total 72 tokens
            fen_embedding = torch.cat(tokens, dim=0) # (72, D)
            batch_tokens.append(fen_embedding)

        # Stack into a batch
        batch_tokens = torch.stack(batch_tokens, dim=0) # (B,72,D)

        # ---7. Repetition Count (1 token) ---
        repetitions = repetitions - 1 # from 1~3 to 0~2
        repetitions = torch.clamp(repetitions,0,2) # if repetition count >3 but no player claimed a draw, it will be treated as 3 repetitions
        repetition_tokens = self.repetition_embed(repetitions) # (B,D)
        repetition_tokens = repetition_tokens.unsqueeze(1) # (B,1,D)

        return torch.cat([batch_tokens,repetition_tokens], dim=1) # (B, 73, D)
