import math
from collections import defaultdict
from itertools import product
from typing import List, Optional, Dict

import chess
import chess.polyglot
import chess.engine
import numpy as np
from reconchess.utilities import slide_move

from functools import lru_cache

def bool_to_color(boolean_value: chess.Color):
    return 'White' if boolean_value else 'Black'


class HashableBoard(chess.Board):
    def __hash__(self):
        return hash(self._transposition_key())


def get_3x3_center_squares(square):
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    ranks = [rank + dr for dr in range(-1, 2) if 0 <= rank + dr <= 7]
    files = [file + df for df in range(-1, 2) if 0 <= file + df <= 7]
    return [chess.square(file, rank) for rank, file in product(ranks, files)]


def get_best_center_from_best_op_moves_dict(best_moves_for_opponent: Dict[chess.Move, float]):
    # Shift the values in the dictionary so the lowest value becomes 0
    min_value = min(best_moves_for_opponent.values())
    shifted_best_moves_for_opponent = {k: (v - min_value) + 0.1 for k, v in best_moves_for_opponent.items()}

    # Find the 3x3 region with most targeted squares
    best_center = None
    best_count = 0
    for square in chess.SQUARES:
        # Exclude squares in the last two ranks for each player
        if chess.square_rank(square) in [0, 7] or chess.square_file(square) in [0, 7]:
            continue
        center_squares = get_3x3_center_squares(square)
        count = sum(
            value
            for move, value in shifted_best_moves_for_opponent.items()
            if (move.to_square in center_squares or move.from_square in center_squares)
        )
        if count > best_count:
            best_count = count
            best_center = square
    return best_center


def convert_centipawn_score_to_win_probability(score: int, k: int = 8) -> float:
    # See https://www.chessprogramming.org/Pawn_Advantage,_Win_Percentage,_and_Elo
    return 1 / (1 + 10 ** (-score / (k * 100)))


def current_mover_gives_check(board: HashableBoard) -> bool:
    king = board.king(not board.turn)
    attackers_mask = board.attackers_mask(board.turn, king)
    return bool(attackers_mask)


def get_resulting_move(board: HashableBoard, move: chess.Move) -> Optional[chess.Move]:
    if move in pseudo_legal_moves_with_castling_through_check(board):
        return move
    piece = board.piece_at(move.from_square)
    if not piece:
        return None
    if piece.piece_type in [chess.PAWN, chess.ROOK, chess.BISHOP, chess.QUEEN]:
        move = slide_move(board, move)
    return move


def compute_softmax_probabilities(values, temperature):
    # Subtract max value for numerical stability
    values = np.array(values) - np.max(values)
    probabilities = np.exp(values / temperature)
    probabilities /= np.sum(probabilities)
    return probabilities


def sample_action(actions, probabilities):
    return np.random.choice(actions, p=probabilities)


def lc0_q_value_to_centipawn_score(q_value: float) -> float:
    return 290.680623072 * np.tan(1.5620688421 * q_value)


def find_best_move_among_all(move_weights: Dict, move_counts: Dict, boards: List[HashableBoard]) -> Optional[chess.Move]:
    # Sort the moves by total weight, highest first.
    sorted_moves = sorted(move_weights, key=move_weights.get, reverse=True)
    if not sorted_moves:
        # We are mated everywhere...
        print("MATED EVERYWHERE")
        print(boards)
        return None
    best_move_uci = sorted_moves[0]
    print(best_move_uci)
    best_move = chess.Move.from_uci(best_move_uci)

    return best_move


def extend_if_possible(best_move: chess.Move, move_weights: Dict, move_counts: Dict, board: HashableBoard) -> chess.Move:
    # Check if move is a sliding move
    piece_to_move = board.piece_at(best_move.from_square)

    # Define the set of sliding piece types
    sliding_pieces = {chess.PAWN, chess.BISHOP, chess.ROOK, chess.QUEEN}
    print(f'Fen: {board.fen()}')
    print(f'Best move: {best_move}')
    if piece_to_move.piece_type in sliding_pieces:
        # Keep trying to extend the move until no further extension is possible or
        # the extended move is not better than the current best move
        while True:
            # Try to extend the move
            extended_move = extend_sliding_move(best_move)
            extended_move_uci = extended_move.uci() if extended_move else None
            # If the extended move is valid and has been made before
            if extended_move_uci and move_counts.get(extended_move_uci, 0) > 0:
                best_move_uci = best_move.uci()
                # If the average weight of the extended move is better than the current best move
                if (move_weights[best_move_uci] / move_counts[best_move_uci]) < (
                        move_weights[extended_move_uci] / move_counts[extended_move_uci]):
                    # Update the best move
                    best_move = extended_move
                    print(f'Extended move: {best_move}')
                else:
                    break
            else:
                break

    return best_move


def convert_castling_moves_if_any(best_move: chess.Move) -> chess.Move:
    # Need to manually convert castling moves...
    if best_move.uci() in ['e8h8', 'e8a8', 'e1h1', 'e1a1']:
        if best_move.uci() == 'e8h8':
            best_move = chess.Move.from_uci('e8g8')
        elif best_move.uci() == 'e8a8':
            best_move = chess.Move.from_uci('e8c8')
        elif best_move.uci() == 'e1h1':
            best_move = chess.Move.from_uci('e1g1')
        elif best_move.uci() == 'e1a1':
            best_move = chess.Move.from_uci('e1c1')

    return best_move


def extend_sliding_move(move: chess.Move) -> Optional[chess.Move]:
    # get the rank and file of the from and to squares
    from_rank, from_file = divmod(move.from_square, 8)
    to_rank, to_file = divmod(move.to_square, 8)

    # calculate the direction of the move in the rank and file dimensions
    rank_direction = (to_rank - from_rank) // max(1, abs(to_rank - from_rank))
    file_direction = (to_file - from_file) // max(1, abs(to_file - from_file))

    # extend the move by one square in the same direction
    extended_to_rank = to_rank + rank_direction
    extended_to_file = to_file + file_direction

    # make sure the extended move is within the board
    if 0 <= extended_to_rank < 8 and 0 <= extended_to_file < 8:
        extended_move = chess.Move(move.from_square, extended_to_rank * 8 + extended_to_file)
    else:
        # if the extended move is off the board, return the original move
        extended_move = move

    return extended_move


def calculate_square_entropy(chess_boards: List[HashableBoard], square: chess.Square) -> float:
    # Initialize a dictionary to count the occurrences of each piece
    counts = defaultdict(int)

    # Iterate over each board and update the counts
    for board in chess_boards:
        piece = board.piece_at(square)
        counts[piece.piece_type if piece is not None else None] += 1

    # Calculate the total count
    total_count = sum(counts.values())

    # Calculate the entropy
    entropy = 0
    for count in counts.values():
        if count > 0:  # Avoid 0 * log(0)
            probability = count / total_count
            entropy -= probability * math.log2(probability)

    return entropy


def move_square_away_from_edges(square: chess.Square) -> chess.Square:
    # Get the rank and file of the square
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    # Move the file and rank away from the edges
    if file == 0:
        file += 1
    elif file == 7:
        file -= 1

    if rank == 0:
        rank += 1
    elif rank == 7:
        rank -= 1

    # Combine the file and rank back into a square
    square = chess.square(file, rank)
    return square


def possible_piece_types_from_move(move: chess.Move) -> List[chess.PieceType]:
    # Converts a move to all pieces that could have executed it
    from_square = move.from_square
    to_square = move.to_square

    diff_file = abs(chess.square_file(from_square) - chess.square_file(to_square))
    diff_rank = abs(chess.square_rank(from_square) - chess.square_rank(to_square))

    possible_pieces = []

    # Check for knight movement
    if (diff_file == 1 and diff_rank == 2) or (diff_file == 2 and diff_rank == 1):
        possible_pieces.append(chess.KNIGHT)
        return possible_pieces

    # Check for diagonal sliding
    if diff_file == diff_rank and diff_file > 0:
        possible_pieces.extend([chess.BISHOP, chess.QUEEN])
        if diff_file == 1:
            possible_pieces.append(chess.KING)
            possible_pieces.append(chess.PAWN)  # Assuming it's a capturing move
        return possible_pieces

    # Check for straight sliding
    if (diff_file == 0 and diff_rank > 0) or (diff_file > 0 and diff_rank == 0):
        possible_pieces.extend([chess.ROOK, chess.QUEEN])
        if diff_file <= 1 and diff_rank <= 1:
            possible_pieces.append(chess.KING)
        if diff_rank == 1 and diff_file == 0:  # Vertical pawn move
            possible_pieces.append(chess.PAWN)
        if diff_rank == 2 and diff_file == 0:  # Pawn initial double step move
            # Check if the pawn is on one of the 16 starting squares
            starting_squares = list(range(8, 16)) + list(range(48, 56))  # squares a2-h2 and a7-h7
            if from_square in starting_squares:
                possible_pieces.append(chess.PAWN)

    # Check for king-side castle
    if from_square == chess.E1 and to_square == chess.G1 or from_square == chess.E8 and to_square == chess.G8:
        possible_pieces.append(chess.KING)

    # Check for queen-side castle
    if from_square == chess.E1 and to_square == chess.C1 or from_square == chess.E8 and to_square == chess.C8:
        possible_pieces.append(chess.KING)

    return possible_pieces


MOVE_CACHE = {}



@lru_cache(maxsize=None)
def pseudo_legal_moves_with_castling_through_check(board: HashableBoard) -> List[chess.Move]:
    # Check if we've already computed the moves for this board
    moves = [chess.Move.null(), *board.generate_pseudo_legal_moves()]

    castling_rights = board.castling_rights
    if castling_rights & chess.BB_H1:
        moves.append(chess.Move(chess.E1, chess.G1))  # White short castle
    if castling_rights & chess.BB_A1:
        moves.append(chess.Move(chess.E1, chess.C1))  # White long castle
    if castling_rights & chess.BB_H8:
        moves.append(chess.Move(chess.E8, chess.G8))  # Black short castle
    if castling_rights & chess.BB_A8:
        moves.append(chess.Move(chess.E8, chess.C8))  # Black long castle
    return moves
