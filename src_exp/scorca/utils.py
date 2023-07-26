import math
from collections import defaultdict
from itertools import product
from typing import List, Optional, Dict, Tuple, Set
import os 
import chess
import chess.engine
import numpy as np
from lczero.backends import Backend, Weights, GameState
from reconchess.utilities import slide_move

from multiprocessing import Pool, Manager
from collections import defaultdict


# VALUE_MATE = 32768
VALUE_MATE = 10000
TIME_USED_FOR_OPERATION = 5

# 24 blocks x 320 filters
T60 = 'weights_run1_814501.lc0'  # 96 sec for 10000 evals

# 20 blocks x 256 filters
LEELENSTEIN = '/Users/robinbux/Desktop/RBC_New/lc0_nets/20x256SE-jj-9-75000000.pb'  # 59 sec for 10000 evals

# 15/16 blocks x 192 filters
T79 = '/Users/robinbux/Desktop/RBC_New/lc0_nets/weights_run2_792013.lc0'  # 51 sec for 10000 evals

# 15 blocks x 768 filters
T1_786 = '/Users/robinbux/Desktop/RBC_New/lc0_nets/t1-768x15x24h-swa-4000000.pb'  # 92 sec for 10000 evals

# 15 blocks x 512 filters
T1_512 = '/Users/robinbux/Desktop/RBC_New/lc0_nets/t1-smolgen-512x15x8h-distilled-swa-3395000.pb'

script_dir = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(script_dir, '..', '..', 'lc0_nets', T60)

# Load weights
w = Weights(weights_path)

# Choose a backend
b = Backend(weights=w)


def bool_to_color(boolean_value: chess.Color):
    return 'White' if boolean_value else 'Black'


def get_best_moves_l0(boards: List[chess.Board], n_likely_boards_per_state: Optional[int] = None) -> Tuple[
    Dict[str, float], Set[chess.Board]]:
    all_moves = set()
    for board in boards:
        all_moves.update(pseudo_legal_moves_with_castling_through_check(board))
    move_weights, move_counts, likely_boards = get_move_weights_and_move_counts(boards, list(all_moves),
                                                                                n_likely_boards_per_state)

    return move_weights, likely_boards


def get_3x3_center_squares(square):
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    ranks = [rank + dr for dr in range(-1, 2) if 0 <= rank + dr <= 7]
    files = [file + df for df in range(-1, 2) if 0 <= file + df <= 7]
    return [chess.square(file, rank) for rank, file in product(ranks, files)]


def get_best_center_from_target_squares(target_squares):
    # Find the 3x3 region with most targeted squares
    best_center = None
    best_count = 0
    for square in chess.SQUARES:
        # Exclude squares in the last two ranks for each player
        if chess.square_rank(square) in [0, 7] or chess.square_file(square) in [0, 7]:
            continue
        center_squares = get_3x3_center_squares(square)
        count = sum(target_squares[sq] for sq in center_squares)
        if count > best_count:
            best_count = count
            best_center = square
    return best_center


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


def mate_to_cp(mate_in_n) -> int:
    # Substracting 1000 centipawns for every move required to mate
    # https://chess.stackexchange.com/questions/22087/how-is-average-centipawn-loss-calculated-when-a-mate-is-missed
    return VALUE_MATE - (mate_in_n * 1000)


def convert_centipawn_score_to_win_probability(score: int, k: int = 8) -> float:
    # See https://www.chessprogramming.org/Pawn_Advantage,_Win_Percentage,_and_Elo
    return 1 / (1 + 10 ** (-score / (k * 100)))


def current_mover_gives_check(board: chess.Board) -> bool:
    king = board.king(not board.turn)
    attackers_mask = board.attackers_mask(board.turn, king)
    return bool(attackers_mask)


def get_resulting_move(board: chess.Board, move: chess.Move, pseudo_legal_moves: List[chess.Move]) -> chess.Move:
    if move in pseudo_legal_moves:
        return move
    piece = board.piece_at(move.from_square)
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


KING_CAPTURE_SCORE = 5
LEAVE_IN_CHECK_SCORE = 10
CHECK_WITHOUT_CAPTURE_SCORE = 3

def lc0_q_value_to_centipawn_score(q_value: float) -> float:
    cp = 290.680623072 * np.tan(1.5620688421 * q_value)
    return cp


def calculate_move_weights_and_get_likely_boards(board: chess.Board, b, possible_moves: List[chess.Move],
                                                 n_likely_boards_to_return: Optional[int] = None) -> Tuple[
    Dict, Dict, Optional[Set[chess.Board]]]:
    move_weights = defaultdict(float)
    move_counts = defaultdict(int)

    # 2 Special cases:
    # 1. If the opponent is in check, we want to capture the king
    # 2. If we are checkmate we skip

    # First case:
    if current_mover_gives_check(board):
        op_king_square = board.king(not board.turn)
        pseudo_legal_moves = pseudo_legal_moves_with_castling_through_check(board)
        for move in pseudo_legal_moves:
            if move.to_square == op_king_square:
                move_weights[move.uci()] += KING_CAPTURE_SCORE
                move_counts[move.uci()] += 1
        return move_weights, move_counts, None

    # Second case:
    if board.is_checkmate():
        return move_weights, move_counts, None

    game_state = GameState(board.fen())
    i2 = game_state.as_input(b)
    eval = b.evaluate(i2)[0]

    q = eval.q()  # q will be in [-1, 1]
    cp = lc0_q_value_to_centipawn_score(q)

    moves = game_state.moves()
    policy_indices = game_state.policy_indices()
    move_scores = list(zip(moves, eval.p_raw(*policy_indices)))  # list of (move, probability) tuples
    move_scores_softmax = list(zip(moves, eval.p_softmax(*policy_indices)))

    sorted_moves = sorted(move_scores_softmax, key=lambda x: x[1], reverse=True)

    likely_boards = set()

    for idx, (move, score) in enumerate(sorted_moves):
        # Get piece of move
        move_weights[move] -= math.log(idx + 1) * 2.5
        move_counts[move] += 1

        move_obj = chess.Move.from_uci(move)

        # If the move does not capture a piece and puts the opponent in check, add a bonus to the move's weight
        if not board.piece_at(move_obj.to_square) and board.gives_check(move_obj):
            move_weights[move] += CHECK_WITHOUT_CAPTURE_SCORE
            move_counts[move] += 1

        if n_likely_boards_to_return and idx < n_likely_boards_to_return:
            likely_board = board.copy()
            likely_board.push(chess.Move.from_uci(move))
            likely_boards.add(likely_board)

    # If we are into check on this board, subtract points for all moves not suggested here
    for possible_move in possible_moves:
        if possible_move.uci() not in move_weights:
            # We have to see WHY we can't make the move. If it is a capture or sliding move, maybe on this board
            # something is just blocking it or there is no piece to capture
            # To check we just 'remove' our king from the board.
            board_copy = board.copy()
            king_pos = board_copy.king(board.turn)
            is_king_move = possible_move.from_square == king_pos
            board_copy.remove_piece_at(king_pos)
            legal_moves = list(board_copy.legal_moves)
            if (possible_move in legal_moves) or is_king_move:
                move_weights[possible_move.uci()] -= LEAVE_IN_CHECK_SCORE
                move_counts[possible_move.uci()] += 1
            else:
                move_weights[possible_move.uci()] -= math.log(len(possible_moves) // 2 + 1) * 2.5
                move_counts[possible_move.uci()] += 1


    return move_weights, move_counts, likely_boards


def find_best_move_among_all(move_weights: Dict, move_counts: Dict, boards: List[chess.Board]) -> Optional[chess.Move]:
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


def extend_if_possible(best_move: chess.Move, move_weights: Dict, move_counts: Dict, board: chess.Board) -> chess.Move:
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


def worker(args):
    board, possible_moves, n_likely_boards_per_state = args
    weights, counts, likely_boards = calculate_move_weights_and_get_likely_boards(board, b, possible_moves,
                                                                                  n_likely_boards_per_state)
    return weights, counts, likely_boards


def get_move_weights_and_move_counts(boards: List[chess.Board], possible_moves: List[chess.Move],
                                     n_likely_boards_per_state: Optional[int] = None) -> Tuple[
    Dict, Dict, Set[chess.Board]]:
    move_weights = defaultdict(float)
    move_counts = defaultdict(int)
    all_likely_boards = set()

    with Pool() as p:
        results = p.map(worker, [(board, possible_moves, n_likely_boards_per_state) for board in boards])

    for weights, counts, likely_boards in results:
        if likely_boards:
            all_likely_boards.update(likely_boards)

        for move, weight in weights.items():
            move_weights[move] += weight

        for move, count in counts.items():
            move_counts[move] += count

    return move_weights, move_counts, all_likely_boards


def find_best_move_l0(boards: List[chess.Board], possible_moves: List[chess.Move]) -> Optional[chess.Move]:
    move_weights, move_counts, _ = get_move_weights_and_move_counts(boards, possible_moves)

    # Find the best move among all moves
    best_move = find_best_move_among_all(move_weights, move_counts, boards)

    # If we are mated everywhere, return None
    if best_move is None:
        return None

    # Extend the best move if it is possible
    best_move = extend_if_possible(best_move, move_weights, move_counts, boards[0])

    print(boards[:20])
    # print(move_weights)
    print(sorted(move_weights.items(), key=lambda item: item[1], reverse=True))

    # Convert the castling moves if the best move is a castling move
    best_move = convert_castling_moves_if_any(best_move)

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


def calculate_square_entropy(chess_boards: List[chess.Board], square: chess.Square) -> float:
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


def pseudo_legal_moves_with_castling_through_check(board: chess.Board) -> List[chess.Move]:
    # Check if we've already computed the moves for this board
    board_fen = board.fen()
    if board_fen in MOVE_CACHE:
        return MOVE_CACHE[board_fen]

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

    MOVE_CACHE[board_fen] = moves
    return moves

