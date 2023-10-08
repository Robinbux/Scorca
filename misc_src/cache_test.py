from functools import lru_cache

import chess

@lru_cache(maxsize=None)
def return_fen(board: chess.Board) -> str:
    x = 5
    v = 3
    return board.fen()

board_one = chess.Board()

board_two = chess.Board()

fen_one = return_fen(board_one)

fen_two = return_fen(board_two)

print(fen_one == fen_two)