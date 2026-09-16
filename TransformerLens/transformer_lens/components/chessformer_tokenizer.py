"""
ChessFormer自定义Tokenizer
基于完整的国际象棋映射和常量
"""

import torch
from typing import List, Dict, Tuple, Set, Any, Optional, Union
from transformers import PreTrainedTokenizerBase, BatchEncoding
import re
import json
import os

# --- Constants --- #
MAX_HALFMOVES = 128  # cap for embedding table size
MAX_FULLMOVES = 256  # cap for embedding table size

# --- Helper Mappings --- #
PIECE_TO_IDX: Dict[str, int] = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    '.': 12
}
IDX_TO_PIECE: Dict[int, str] = {v: k for k, v in PIECE_TO_IDX.items()}
EMPTY_SQ_IDX = PIECE_TO_IDX['.']

# Map algebraic square notation (e.g., 'a1', 'h8') to 0-63 index
# a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
SQUARE_TO_IDX: Dict[str, int] = {
    f"{file}{rank}": (rank - 1) * 8 + (ord(file) - ord('a'))
    for rank in range(1, 9)
    for file in 'abcdefgh'
}
IDX_TO_SQUARE: Dict[int, str] = {v: k for k, v in SQUARE_TO_IDX.items()}

# --- Coordinate and Notation Helpers ---
# Precompute maps for efficiency
_IDX_TO_COORDS: Dict[int, Tuple[int, int]] = {i: (i // 8, i % 8) for i in range(64)}  # (rank, file) 0-7
_COORDS_TO_IDX: Dict[Tuple[int, int], int] = {v: k for k, v in _IDX_TO_COORDS.items()}
_IDX_TO_ALG: Dict[int, str] = {
    i: f"{chr(ord('a') + file)}{rank + 1}"
    for i, (rank, file) in _IDX_TO_COORDS.items()
}
_ALG_TO_IDX: Dict[str, int] = {v: k for k, v in _IDX_TO_ALG.items()}

def _coords_to_alg(r: int, f: int) -> str:
    """Converts 0-indexed (rank, file) to algebraic notation."""
    if 0 <= r < 8 and 0 <= f < 8:
        return f"{chr(ord('a') + f)}{r + 1}"
    # This should not happen with valid indices, but good for safety
    raise ValueError(f"Invalid coordinates: ({r}, {f})")

def generate_structurally_valid_move_map() -> Dict[str, int]:
    """
    Generates a dictionary mapping chess moves that are geometrically possible
    by *some* standard piece (K, Q, R, B, N, or P) to unique integer indices.
    It excludes moves that are structurally impossible for any piece to make
    in one turn (e.g., a1->h5 for non-knight).

    Includes standard UCI promotions (e.g., "e7e8q"), replacing the
    corresponding simple pawn move to the final rank (e.g., "e7e8").
    This is based purely on piece movement geometry, not the current board state.

    Returns:
        Dict[str, int]: A map from the valid UCI move string to a unique
                        integer index (0 to N-1). The size N is expected
                        to be around 1800-1900.
    """
    valid_moves: Set[str] = set()
    # Keep track of base moves (like 'e7e8') that are replaced by promotions
    # according to UCI standard.
    promo_base_moves_to_exclude: Set[str] = set()

    # 1. Generate all geometrically possible non-promotion moves
    for from_idx in range(64):
        from_r, from_f = _IDX_TO_COORDS[from_idx]
        from_alg = _IDX_TO_ALG[from_idx]

        for to_idx in range(64):
            if from_idx == to_idx:
                continue

            to_r, to_f = _IDX_TO_COORDS[to_idx]
            to_alg = _IDX_TO_ALG[to_idx]
            dr, df = to_r - from_r, to_f - from_f
            abs_dr, abs_df = abs(dr), abs(df)

            # Check if the geometry matches any standard piece movement
            # Note: Queen moves are covered by Rook + Bishop checks.
            # Note: Pawn single pushes/captures are covered by King/Rook/Bishop geometry.
            # Note: Pawn double pushes are covered by Rook geometry.
            is_king_move = max(abs_dr, abs_df) == 1
            is_knight_move = (abs_dr == 2 and abs_df == 1) or (abs_dr == 1 and abs_df == 2)
            is_rook_move = dr == 0 or df == 0  # Includes King horiz/vert & pawn double push
            is_bishop_move = abs_dr == abs_df  # Includes King diagonal & pawn capture/push

            if is_king_move or is_knight_move or is_rook_move or is_bishop_move:
                uci_move = f"{from_alg}{to_alg}"
                valid_moves.add(uci_move)

    # 2. Generate promotion moves explicitly and mark base moves for exclusion
    promo_pieces = ['q', 'r', 'b', 'n']
    for from_f in range(8):
        # White promotions (from rank 7 (idx 6) to rank 8 (idx 7))
        from_r_w, to_r_w = 6, 7
        if from_r_w != 7:  # Ensure we are on the correct rank before promotion
            from_alg_w = _coords_to_alg(from_r_w, from_f)
            # Possible destinations: push (df=0), capture left (df=-1), capture right (df=1)
            for df in [-1, 0, 1]:
                to_f_w = from_f + df
                if 0 <= to_f_w < 8:
                    to_alg_w = _coords_to_alg(to_r_w, to_f_w)
                    base_move = f"{from_alg_w}{to_alg_w}"
                    # promo_base_moves_to_exclude.add(base_move) # Mark e.g. "e7e8" for exclusion
                    for p in promo_pieces:
                        valid_moves.add(f"{base_move}{p}")  # Add e.g. "e7e8q"

        # Black promotions (from rank 2 (idx 1) to rank 1 (idx 0))
        from_r_b, to_r_b = 1, 0
        if from_r_b != 0:  # Ensure we are on the correct rank before promotion
            from_alg_b = _coords_to_alg(from_r_b, from_f)
            # Possible destinations: push (df=0), capture left (df=-1), capture right (df=1)
            for df in [-1, 0, 1]:
                to_f_b = from_f + df
                if 0 <= to_f_b < 8:
                    to_alg_b = _coords_to_alg(to_r_b, to_f_b)
                    base_move = f"{from_alg_b}{to_alg_b}"
                    # promo_base_moves_to_exclude.add(base_move) # Mark e.g. "e2e1" for exclusion
                    for p in promo_pieces:
                        valid_moves.add(f"{base_move}{p}")  # Add e.g. "e2e1q"

    # 3. Remove the base moves that were replaced by promotions
    final_valid_moves = valid_moves - promo_base_moves_to_exclude

    # 4. Add draw claim
    final_valid_moves.add("<claim_draw>")

    # 5. Create the final map with sorted keys for deterministic indices
    sorted_moves = sorted(list(final_valid_moves))
    move_map = {move: i for i, move in enumerate(sorted_moves)}

    return move_map

# 生成移动映射
UCI_MOVE_TO_IDX = generate_structurally_valid_move_map()
IDX_TO_UCI_MOVE = {v: k for k, v in UCI_MOVE_TO_IDX.items()}

class ChessFormerTokenizer(PreTrainedTokenizerBase):
    """ChessFormer自定义Tokenizer - 继承自PreTrainedTokenizerBase"""
    
    def __init__(self, **kwargs):
        # 设置基本属性
        self.model_max_length = 75  # ChessFormer的上下文长度
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        
        # 设置特殊token的ID
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        
        # 创建词汇表映射
        self._create_vocab()
        
        # 设置词汇表大小
        self._vocab_size = len(self.vocab)
        
        # 添加TransformerLens需要的属性
        self.name_or_path = "chessformer"
        self.add_bos_token = True  # 直接设置为True，避免重新初始化
        
        # 调用父类初始化
        super().__init__(
            pad_token=self.pad_token,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            unk_token=self.unk_token,
            model_max_length=self.model_max_length,
            **kwargs
        )
        
        # 添加init_kwargs以兼容HuggingFace格式
        self.init_kwargs = {
            "name_or_path": self.name_or_path,
            "add_bos_token": self.add_bos_token,
            "model_max_length": self.model_max_length,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
            "vocab_size": self._vocab_size,
        }
    
    def _create_vocab(self):
        """创建词汇表"""
        self.vocab = {}
        self.ids_to_tokens = {}
        
        # 添加特殊token
        special_tokens = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            self.bos_token: self.bos_token_id,
            self.unk_token: self.unk_token_id,
        }
        
        self.vocab.update(special_tokens)
        self.ids_to_tokens.update({v: k for k, v in special_tokens.items()})
        
        # 添加移动token（从4开始，因为0-3是特殊token）
        for i, move in enumerate(IDX_TO_UCI_MOVE.values()):
            token_id = i + 4
            self.vocab[move] = token_id
            self.ids_to_tokens[token_id] = move
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return self._vocab_size
    
    @property
    def added_tokens_decoder(self) -> Dict[int, str]:
        """返回添加的token解码器"""
        # 对于ChessFormer，我们没有额外的添加token，返回空字典
        return {}
    
    @property
    def is_fast(self) -> bool:
        """是否为快速tokenizer"""
        return False
    
    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """返回添加的token编码器"""
        # 对于ChessFormer，我们没有额外的添加token，返回空字典
        return {}
    
    def get_vocab(self):
        return self.vocab.copy()
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """将FEN字符串tokenize为token列表"""
        # 对于ChessFormer，FEN字符串会被ChessFormerEmbed处理
        # 这里我们返回一个占位符token列表
        return [self.bos_token, self.eos_token]
    
    def _convert_token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        return self.vocab.get(token, self.unk_token_id)
    
    def _convert_id_to_token(self, index: int) -> str:
        """将ID转换为token"""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """将token列表转换为ID列表"""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        
        return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """将ID列表转换为token列表"""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in [self.pad_token_id, self.eos_token_id, self.bos_token_id, self.unk_token_id]:
                continue
            tokens.append(self._convert_id_to_token(token_id))
        
        return tokens
    
    def encode(self, text, **kwargs) -> torch.Tensor:
        """编码FEN字符串为token ID列表"""
        tokens = self._tokenize(str(text))
        ids = self.convert_tokens_to_ids(tokens)
        
        # 确保ids是列表类型
        if isinstance(ids, int):
            ids = [ids]
        
        # 添加特殊token
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        # 返回torch.Tensor而不是列表
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, token_ids, **kwargs) -> str:
        """解码token ID列表为字符串"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        return " ".join(tokens)
    
    def batch_encode_plus(self, batch_text_or_text_pairs, **kwargs) -> BatchEncoding:
        """批量编码"""
        if isinstance(batch_text_or_text_pairs, str):
            batch_text_or_text_pairs = [batch_text_or_text_pairs]
        
        input_ids = []
        attention_mask = []
        
        for text in batch_text_or_text_pairs:
            ids = self.encode(text)
            input_ids.append(ids)
            attention_mask.append(torch.ones(len(ids), dtype=torch.long))
        
        # 将input_ids转换为torch.Tensor
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        
        return BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
    
    def __call__(self, text, **kwargs) -> BatchEncoding:
        """调用方法"""
        return self.batch_encode_plus(text, **kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存tokenizer"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存词汇表
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        config = {
            "model_type": "chessformer",
            "vocab_size": self.vocab_size,
            "model_max_length": self.model_max_length,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return (save_directory,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """从预训练模型加载"""
        return cls(**kwargs)
    
    # 添加PreTrainedTokenizerBase要求的抽象方法
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """构建包含特殊token的输入"""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
    
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        """获取特殊token的mask"""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """创建token类型ID"""
        if token_ids_1 is not None:
            return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)
        return [0] * (len(token_ids_0) + 2)

def create_chessformer_tokenizer():
    """创建ChessFormer tokenizer实例"""
    return ChessFormerTokenizer()

# 测试tokenizer
if __name__ == "__main__":
    tokenizer = create_chessformer_tokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"特殊token: {list(tokenizer.vocab.keys())[:10]}...")
    
    # 测试编码解码
    test_text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"测试文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    
    # 测试移动映射
    print(f"\n移动映射大小: {len(UCI_MOVE_TO_IDX)}")
    print(f"前10个移动: {list(UCI_MOVE_TO_IDX.keys())[:10]}")
    print(f"后10个移动: {list(UCI_MOVE_TO_IDX.keys())[-10:]}") 