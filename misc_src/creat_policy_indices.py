import os
import chess
import chess.engine
import chess.polyglot
import json
from lczero.backends import Backend, Weights, GameState


T60 = 'weights_run2_792013.lc0'  # 96 sec for 10000 evals

script_dir = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(script_dir, '..', 'lc0_nets', T60)
# Load weights
L0_WEIGHTS = Weights(weights_path)
L0_BACKEND = Backend(weights=L0_WEIGHTS)

chess.Board.__hash__ = lambda self: chess.polyglot.zobrist_hash(self)

boards = []


def find_closest_not_in_set(s):
    return next(i for i in range(64) if i not in s)

policy_indices_dict = {}
# Loop through all the ranks and files
for rank in range(8):
    for file in range(8):
        # Queen
        board = chess.Board(None)
        queen_square = chess.square(file, rank)
        board.set_piece_at(queen_square, chess.Piece(chess.QUEEN, chess.WHITE))

        unplaceable_squares = set(board.attacks(queen_square))
        unplaceable_squares.update({queen_square})

        black_king_square = find_closest_not_in_set(unplaceable_squares)
        board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))

        black_king_attacks = set(board.attacks(black_king_square))
        unplaceable_squares.update(black_king_attacks)
        unplaceable_squares.update({black_king_square})

        white_king_square = find_closest_not_in_set(unplaceable_squares)
        board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))

        game_state = GameState(board.fen())
        i2 = game_state.as_input(L0_BACKEND)
        eval = L0_BACKEND.evaluate(i2)[0]

        moves = game_state.moves()
        policy_indices = game_state.policy_indices()

        # For Queen
        for move, index in zip(moves, policy_indices):
            policy_indices_dict[index] = move

        # Knight
        board = chess.Board(None)
        knight_square = chess.square(file, rank)
        board.set_piece_at(knight_square, chess.Piece(chess.KNIGHT, chess.WHITE))

        unplaceable_squares = set(board.attacks(knight_square))
        unplaceable_squares.update({knight_square})

        black_king_square = find_closest_not_in_set(unplaceable_squares)
        board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))

        black_king_attacks = set(board.attacks(black_king_square))
        unplaceable_squares.update(black_king_attacks)
        unplaceable_squares.update({black_king_square})

        white_king_square = find_closest_not_in_set(unplaceable_squares)
        board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))

        game_state = GameState(board.fen())
        i2 = game_state.as_input(L0_BACKEND)
        eval = L0_BACKEND.evaluate(i2)[0]

        moves = game_state.moves()
        policy_indices = game_state.policy_indices()

        for move, index in zip(moves, policy_indices):
            policy_indices_dict[index] = move

def write_dict_to_file(d, filename):
    with open(filename, 'w') as f:
        json.dump(d, f)

write_dict_to_file(policy_indices_dict, "policy_indices.json")