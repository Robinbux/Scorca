import chess
import numpy as np
from functools import lru_cache

from src.rbc_sunfish.sunfish.sunfish import parse, Position, Searcher, render
from src.rbc_sunfish.sunfish.sunfish_nnue import features, layer1, layer2, model


def sunfish_index_to_chess_square(idx):
    if not (20 <= idx <= 99) or idx % 10 == 9 or idx % 10 == 0:
        raise ValueError("Invalid index")

    row = 9 - (idx // 10)
    col = idx % 10 - 1
    square = chess.square(col, row)
    # name = chess.square_name(square)
    return square


def sunfish_move_to_chess_move(sunfish_move, color: chess.Color):
    #print(f'sunfish_move: {sunfish_move}')
    from_square = sunfish_index_to_chess_square(sunfish_move.i)
    #print(f'from_square: {from_square}')
    to_square = sunfish_index_to_chess_square(sunfish_move.j)
    #print(f'to_square: {to_square}')
    promotion = sunfish_move.prom

    if color == chess.BLACK:
        #print('flipping')
        from_square = flip_chess_index(from_square)
        #print(f'from_square flipped: {from_square}')
        to_square = flip_chess_index(to_square)
        #print(f'to_square flipped: {to_square}')

    # Convert the promotion piece to python-chess format
    if promotion:
        promotion = {'N': chess.KNIGHT, 'B': chess.BISHOP, 'R': chess.ROOK, 'Q': chess.QUEEN}[promotion]
    else:
        promotion = None

    return chess.Move(from_square, to_square, promotion=promotion)


def flip_chess_index(index):
    if not (0 <= index < 64):
        raise ValueError("Index must be between 0 and 63.")

    row, col = divmod(index, 8)
    flipped_row = 7 - row
    flipped_col = 7 - col
    return flipped_row * 8 + flipped_col

@lru_cache(maxsize=None)
def from_fen(fen: str, color: chess.Color):
    """
    Convert a FEN string to a Position instance.
    """
    # Split the FEN string into its components
    parts = fen.split()
    board, turn, castling, ep_square = parts[0], parts[1], parts[2], parts[3]

    # Convert the board part of the FEN string to the internal board representation
    rows = board.split('/')
    board = '         \n         \n'
    for row in rows:
        board += ' '
        for c in row:
            board += '.' * int(c) if c.isdigit() else c
        board += '\n'
    board = board + '         \n         \n'

    # Convert the castling part of the FEN string to the internal castling rights representation
    wc = (castling.find('K') > -1, castling.find('Q') > -1)
    bc = (castling.find('k') > -1, castling.find('q') > -1)

    # Convert the en passant square part of the FEN string to the internal en passant square representation
    ep = parse(ep_square) if ep_square != '-' else 0

    wf, bf = features(board)

    score = custom_score_calc(wf, bf)

    # Create and return the Position instance
    pos = Position(board, score, wc, bc, ep, 0)
    if turn == 'b':
        pos = pos.rotate()
    return pos

def custom_score_calc(wf, bf):
    act = np.tanh
    hidden = (layer1[:, :9] @ act(wf[1:])) + (layer1[:, 9:] @ act(bf[1:]))
    score = layer2 @ act(hidden)
    return int((score + model["scale"] * (wf[0] - bf[0])) * 360)
