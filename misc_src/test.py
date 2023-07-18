import collections
from itertools import product
import hashlib

import chess
import chess.engine
import chess.polyglot

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, setpgrp=True)
stockfish_engine.configure({
    "Use NNUE": True,
    'Hash': 2048,
    'Threads': 3,
})

POSITIONS = [
    '5r2/1p3pbk/1p1p1p2/7p/1r3P2/3B2P1/P1P4P/2KN3R b - - 0 27',
    '1r2q2k/4P2p/p2p2p1/2p5/2Pp4/1P6/P3Q1PP/R5K1 b - - 0 25'
]

def hash_chess_board(board):
    # Step 1: Serialize the state of the board. The FEN representation can be used for this.
    fen = board.fen()

    # Step 2: Hash the serialized data. Here I am using SHA-256, but you can use another hash function if you prefer.
    return hashlib.sha256(fen.encode()).hexdigest()



BOARDS = [chess.Board(position) for position in POSITIONS]

board = BOARDS[0]
chess.Board.__hash__ = lambda self: hash(self._transposition_key())
chess.Board.__hash__ = lambda self: chess.polyglot.zobrist_hash(self)

board_hash = hash(board)
new_board_hash = hash_chess_board(board)
zobrist_hash = chess.polyglot.zobrist_hash(board)

print(board.is_valid())
