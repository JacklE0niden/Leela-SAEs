import chess
import numpy as np
import random
from typing import Sequence, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

'''
该文件的目的是把输入字符串 f{fen,action} 转换成 tokens
'''

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

# pyfmt: disable
_CHARACTERS = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'p',
    'n',
    'r',
    'k',
    'q',
    'P',
    'B',
    'N',
    'R',
    'Q',
    'K',
    'w',
    '.',
]
# pyfmt: enable
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 77


def tokenize(fen: str) -> np.ndarray:
    """Returns an array of tokens from a fen string.

    We compute a tokenized representation of the board, from the FEN string.
    The final array of tokens is a mapping from this string to numbers, which
    are defined in the dictionary `_CHARACTERS_INDEX`.
    For the 'en passant' information, we convert the '-' (which means there is
    no en passant relevant square) to '..', to always have two characters, and
    a fixed length output.

    Args:
        fen: The board position in Forsyth-Edwards Notation.
    """
    # Extracting the relevant information from the FEN.
    board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
    board = board.replace('/', '')
    board = side + board

    indices = list()

    for char in board:
        if char in _SPACES_CHARACTERS:
            indices.extend(int(char) * [_CHARACTERS_INDEX['.']])
        else:
            indices.append(_CHARACTERS_INDEX[char])

    if castling == '-':
        indices.extend(4 * [_CHARACTERS_INDEX['.']])
    else:
        for char in castling:
            indices.append(_CHARACTERS_INDEX[char])
        # Padding castling to have exactly 4 characters.
        if len(castling) < 4:
            indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

    if en_passant == '-':
        indices.extend(2 * [_CHARACTERS_INDEX['.']])
    else:
        # En passant is a square like 'e3'.
        for char in en_passant:
            indices.append(_CHARACTERS_INDEX[char])

    # Three digits for halfmoves (since last capture) is enough since the game
    # ends at 50.
    halfmoves_last += '.' * (3 - len(halfmoves_last))
    indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

    # Three digits for full moves is enough (no game lasts longer than 999
    # moves).
    fullmoves += '.' * (3 - len(fullmoves))
    indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

    assert len(indices) == SEQUENCE_LENGTH

    return np.asarray(indices, dtype=np.uint8)



_CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def _compute_all_possible_actions() -> tuple[dict[str, int], dict[int, str]]:
  """Returns two dicts converting moves to actions and actions to moves.

  These dicts contain all possible chess moves.
  """
  all_moves = []

  # First, deal with the normal moves.
  # Note that this includes castling, as it is just a rook or king move from one
  # square to another.
  board = chess.BaseBoard.empty()
  for square in range(64):
    next_squares = []

    # Place the queen and see where it attacks (we don't need to cover the case
    # for a bishop, rook, or pawn because the queen's moves includes all their
    # squares).
    board.set_piece_at(square, chess.Piece.from_symbol('Q'))
    next_squares += board.attacks(square)

    # Place knight and see where it attacks
    board.set_piece_at(square, chess.Piece.from_symbol('N'))
    next_squares += board.attacks(square)
    board.remove_piece_at(square)

    for next_square in next_squares:
      all_moves.append(
          chess.square_name(square) + chess.square_name(next_square)
      )

  # Then deal with promotions.
  # Only look at the last ranks.
  promotion_moves = []
  for rank, next_rank in [('2', '1'), ('7', '8')]:
    for index_file, file in enumerate(_CHESS_FILE):
      # Normal promotions.
      move = f'{file}{rank}{file}{next_rank}'
      promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

      # Capture promotions.
      # Left side.
      if file > 'a':
        next_file = _CHESS_FILE[index_file - 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
      # Right side.
      if file < 'h':
        next_file = _CHESS_FILE[index_file + 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
  all_moves += promotion_moves

  move_to_action, action_to_move = {}, {}
  for action, move in enumerate(all_moves):
    assert move not in move_to_action
    move_to_action[move] = action
    action_to_move[action] = move

  return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_actions()

def get_ordered_legal_moves(board: chess.Board) -> Sequence[chess.Move]:
  """Returns legal moves ordered by action value."""
  return sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION[x.uci()])




# # 构造输入tokens  
# def analyse(fen: str, action: str):
# """Returns buckets log-probs for each action, and FEN."""
# # Tokenize the legal actions.
# board = chess.Board(fen)
# if action is not None:
#     if action in MOVE_TO_ACTION:
#         action_idx = MOVE_TO_ACTION[action]
#         print("action_idx:", action_idx)
#     else:
#         raise ValueError(f"Action {action} is not a possible move")
# else:
#     sorted_legal_moves = get_ordered_legal_moves(board)
#     # random choose one move
#     action_idx = random.choice(sorted_legal_moves)

# tokenized_fen = tokenize(board.fen()).astype(np.int32)
# sequences = np.concatenate([tokenized_fen, action_idx, np.zeros((1, 1))], axis=1)
# print("sequences.shape:", sequences.shape)
# # sequences = np.stack([tokenized_fen] * len(legal_actions))
# # Create the sequences.

# # print("sequences.shape:", sequences.shape) # [35, 79]
# return sequences, board.fen()
# return {'log_probs': self.predict_fn(sequences)[:, -1], 'fen': board.fen()}



# SearchlessChessTokenizer
class SearchlessChessTokenizer(nn.Module):
    """
    用于将 FEN+action 转换为模型输入 tokens 的工具类。
    """

    def __init__(self):
        super().__init__()
        self.move_to_action, self.action_to_move = self._compute_all_possible_actions()
        self.padding_side = "right"  # 关键：加上这一行

    @staticmethod
    def _compute_all_possible_actions() -> Tuple[dict[str, int], dict[int, str]]:
        """返回所有可能走子的映射表。"""
        all_moves = []
        _CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        board = chess.BaseBoard.empty()
        for square in range(64):
            next_squares = []
            board.set_piece_at(square, chess.Piece.from_symbol('Q'))
            next_squares += board.attacks(square)
            board.set_piece_at(square, chess.Piece.from_symbol('N'))
            next_squares += board.attacks(square)
            board.remove_piece_at(square)
            for next_square in next_squares:
                all_moves.append(
                    chess.square_name(square) + chess.square_name(next_square)
                )

        # Promotion moves
        promotion_moves = []
        for rank, next_rank in [('2', '1'), ('7', '8')]:
            for index_file, file in enumerate(_CHESS_FILE):
                move = f'{file}{rank}{file}{next_rank}'
                promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
                if file > 'a':
                    next_file = _CHESS_FILE[index_file - 1]
                    move = f'{file}{rank}{next_file}{next_rank}'
                    promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
                if file < 'h':
                    next_file = _CHESS_FILE[index_file + 1]
                    move = f'{file}{rank}{next_file}{next_rank}'
                    promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
        all_moves += promotion_moves

        move_to_action, action_to_move = {}, {}
        for action, move in enumerate(all_moves):
            assert move not in move_to_action
            move_to_action[move] = action
            action_to_move[action] = move

        return move_to_action, action_to_move

    def get_ordered_legal_moves(self, board: chess.Board) -> Sequence[chess.Move]:
        """返回按action索引排序的合法走子。"""
        return sorted(board.legal_moves, key=lambda x: self.move_to_action[x.uci()])

    def tokenize(self, fen: str) -> np.ndarray:
        """FEN字符串转token序列。"""
        # ... 这里直接复用你的tokenize实现 ...
        board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
        board = board.replace('/', '')
        board = side + board

        indices = []
        for char in board:
            if char in _SPACES_CHARACTERS:
                indices.extend(int(char) * [_CHARACTERS_INDEX['.']])
            else:
                indices.append(_CHARACTERS_INDEX[char])

        if castling == '-':
            indices.extend(4 * [_CHARACTERS_INDEX['.']])
        else:
            for char in castling:
                indices.append(_CHARACTERS_INDEX[char])
            if len(castling) < 4:
                indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

        if en_passant == '-':
            indices.extend(2 * [_CHARACTERS_INDEX['.']])
        else:
            for char in en_passant:
                indices.append(_CHARACTERS_INDEX[char])

        halfmoves_last += '.' * (3 - len(halfmoves_last))
        indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])
        fullmoves += '.' * (3 - len(fullmoves))
        indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

        assert len(indices) == SEQUENCE_LENGTH
        return np.asarray(indices, dtype=np.uint8)

    def forward(
        self,
        input: Union[str, List[str]],
        action: Optional[str] = None,
        random_if_none: bool = True
    ) -> Tuple[torch.Tensor, Union[str, List[str]]]:
        """
        将 FEN 和 action 转换为模型输入 tokens。
        Args:
            input: 输入格式为f{fen','action}，可以是单个字符串或字符串列表
            action: 走子字符串（如 'e2e4'），如果为None则随机选一个合法走子
            random_if_none: action为None时是否随机选一个合法走子
        Returns:
            sequences: [batch_size, 79] 的 torch 张量（77个FEN token + 1个action索引 + 1个dummy_return_buckets）
            fen: 标准化后的FEN字符串或字符串列表
        """
        # 处理单个字符串输入
        if isinstance(input, str):
            return self._forward_single(input, action, random_if_none)
        
        # 处理字符串列表输入
        batch_sequences = []
        batch_fens = []
        
        for single_input in input:
            sequences, fen = self._forward_single(single_input, action, random_if_none)
            batch_sequences.append(sequences)
            batch_fens.append(fen)
        
        # 拼接所有序列
        batch_tensor = torch.cat(batch_sequences, dim=0)
        return batch_tensor, batch_fens
    
    def _forward_single(
        self,
        input: str,
        action: Optional[str] = None,
        random_if_none: bool = True
    ) -> Tuple[torch.Tensor, str]:
        """
        处理单个输入字符串的辅助方法。
        """
        fen, action = input.split(',')
        fen = fen.strip()
        action = action.strip()
        
        board = chess.Board(fen)
        if action is not None:
            if action in self.move_to_action:
                action_idx = self.move_to_action[action]
            else:
                raise ValueError(f"Action {action} is not a possible move")
        else:
            sorted_legal_moves = self.get_ordered_legal_moves(board)
            if not sorted_legal_moves:
                raise ValueError("No legal moves available")
            action_idx = self.move_to_action[sorted_legal_moves[0].uci()] if not random_if_none \
                else self.move_to_action[random.choice(sorted_legal_moves).uci()]

        tokenized_fen = self.tokenize(board.fen()).astype(np.int32)
        # 拼接action索引
        sequences = np.concatenate([tokenized_fen, np.array([action_idx], dtype=np.int32)])
        sequences = np.expand_dims(sequences, axis=0)
        # print("sequences.shape:", sequences.shape)
        dummy_return_buckets = np.zeros((1, 1), dtype=np.int32)
        sequences = np.concatenate([sequences, dummy_return_buckets], axis=1)
        sequences = torch.tensor(sequences)
        # sequences = sequences.reshape(1, -1)  # [1, 79]
        return sequences, board.fen()
    
    def encode(self, input: str) -> np.ndarray:
        sequences, _ = self.forward(input)
        return sequences
    
    def decode(self, tokens: np.ndarray) -> str:
        """
        将token序列转换为字符串表示。
        
        Args:
            tokens: shape为[1, 79]的numpy数组，包含77个FEN token + 1个action索引 + 1个dummy_return_buckets
            
        Returns:
            str: 解码后的字符串，格式为"FENtokens+move+dummy"，所有字符均为token集合中的字符
        """
        if tokens.ndim == 2:
            tokens = tokens[0]  # 去掉batch维度
        
        assert len(tokens) == 79, f"Expected 79 tokens, got {len(tokens)}"
        
        # 分离FEN tokens、action token和dummy token
        fen_tokens = tokens[:77]  # 前77个是FEN tokens
        action_token = tokens[77]  # 第78个是action索引
        dummy_token = tokens[78]  # 第79个是dummy_return_buckets
        
        # 解码FEN部分
        fen_str = self._decode_fen_tokens(fen_tokens)
        
        # 解码action部分
        if int(action_token) in self.action_to_move:
            move_str = self.action_to_move[int(action_token)]
        else:
            move_str = f"<unknown_action_{action_token}>"
        
        # 解码dummy部分
        # dummy_token通常为0，这里直接转为字符串
        dummy_str = str(int(dummy_token))
        
        # 组合结果：直接拼接，不加任何分隔符
        result = f"{fen_str}{move_str}{dummy_str}"
        return result
    
    def _decode_fen_tokens(self, fen_tokens: np.ndarray) -> str:
        """
        将FEN tokens解码为原始token字符序列（不做任何分隔和格式化）。
        
        Args:
            fen_tokens: 77个FEN token的数组
            
        Returns:
            str: 77字符的字符串，每个字符均为token集合中的字符
        """
        index_to_char = {v: k for k, v in _CHARACTERS_INDEX.items()}
        chars = []
        for token in fen_tokens:
            if int(token) in index_to_char:
                chars.append(index_to_char[int(token)])
            else:
                # 若遇到未知token，直接用'.'占位
                chars.append('.')
        fen_str = ''.join(chars)
        return fen_str


class SearchlessChessHFTokenizer(PreTrainedTokenizer):
    """
    HuggingFace风格的国际象棋FEN+action tokenizer。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 你可以把 SearchlessChessTokenizer 的初始化内容搬过来
        self.move_to_action, self.action_to_move = SearchlessChessTokenizer._compute_all_possible_actions()
        self.vocab = _CHARACTERS_INDEX
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.model_max_length = SEQUENCE_LENGTH + 2  # 77 + action + dummy
        self.padding_side = "right"  # 关键：加上这一行

    def _tokenize(self, text: str) -> list[str]:
        # 只做单个FEN字符串的tokenize
        tokens = SearchlessChessTokenizer().tokenize(text)
        return [str(t) for t in tokens]

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        # 这里可以实现为FEN字符串的还原（可选）
        return " ".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def __call__(self, fen: str, action: str = None, return_tensors=None, **kwargs):
        # 兼容HF的 __call__ 接口
        tokenizer = SearchlessChessTokenizer()
        tokens, normed_fen = tokenizer.forward(fen, action)
        if return_tensors == "np":
            return {"input_ids": tokens}
        elif return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor(tokens, dtype=torch.long)}
        else:
            return {"input_ids": tokens.tolist()}



