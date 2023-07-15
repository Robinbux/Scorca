import math
import os

from collections import defaultdict
from multiprocessing import Pool
from typing import List, Optional, Tuple, Dict, Set, Any
from lczero.backends import Backend, Weights, GameState


import chess.engine
import chess.polyglot

from utils import convert_castling_moves_if_any, extend_if_possible, find_best_move_among_all, \
    current_mover_gives_check, pseudo_legal_moves_with_castling_through_check, lc0_q_value_to_centipawn_score

NULL_MOVE = chess.Move.null()

# VALUE_MATE = 32768
VALUE_MATE = 10000
TIME_USED_FOR_OPERATION = 5

KING_CAPTURE_SCORE = 5
LEAVE_IN_CHECK_SCORE = 10
CHECK_WITHOUT_CAPTURE_SCORE = 3

# l0 Network weights
# 24 blocks x 320 filters
T60 = 'weights_run1_814174.lc0'  # 96 sec for 10000 evals
# 20 blocks x 256 filters
LEELENSTEIN = '.20x256SE-jj-9-75000000.pb'  # 59 sec for 10000 evals
# 15/16 blocks x 192 filters
T79 = 'weights_run2_792013.lc0'  # 51 sec for 10000 evals
# 15 blocks x 768 filters
T1_786 = 't1-768x15x24h-swa-4000000.pb'  # 92 sec for 10000 evals
# 15 blocks x 512 filters
T1_512 = 't1-smolgen-512x15x8h-distilled-swa-3395000.pb'

script_dir = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(script_dir, '..', '..', 'lc0_nets', T60)

# Load weights
L0_WEIGHTS = Weights(weights_path)
L0_BACKEND = Backend(weights=L0_WEIGHTS)

chess.Board.__hash__ = lambda self: chess.polyglot.zobrist_hash(self)

class MoveStrategy:
    def __init__(self, game_information_db, logger):
        self.logger = logger
        self.game_information_db = game_information_db

    def move(self, move_actions: List[chess.Move], states: Set[chess.Board], seconds_left: int) -> Tuple[Optional[
        chess.Move], List[chess.Move]]:
        self.move_actions = move_actions
        # TODO: Some form of time management....
        return self.leela_chess_zero_move(states)

    def leela_chess_zero_move(self, possible_boards: List[chess.Board]) -> Tuple[
        Optional[chess.Move], List[chess.Move]]:
        possible_boards = list(possible_boards)
        best_move = self.find_best_move_l0(possible_boards, self.move_actions)
        return best_move, []

    def get_best_moves_l0(self, boards: List[chess.Board], n_likely_boards_per_state: Optional[int] = None) -> Tuple[
        Dict[str, float], Set[chess.Board]]:
        all_moves = set()
        for board in boards:
            all_moves.update(pseudo_legal_moves_with_castling_through_check(board))
        move_weights, move_counts, likely_boards = self.get_move_weights_and_move_counts(boards, list(all_moves),
                                                                                    n_likely_boards_per_state)

        return move_weights, likely_boards

    def find_best_move_l0(self, boards: List[chess.Board], possible_moves: List[chess.Move]) -> Optional[chess.Move]:
        move_weights, move_counts, _ = self.get_move_weights_and_move_counts(boards, possible_moves)

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



    def get_move_weights_and_move_counts(self, boards: List[chess.Board], possible_moves: List[chess.Move],
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

    @staticmethod
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
        #cp = lc0_q_value_to_centipawn_score(q)

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

def worker(args):
    board, possible_moves, n_likely_boards_per_state = args
    weights, counts, likely_boards = MoveStrategy.calculate_move_weights_and_get_likely_boards(board, L0_BACKEND, possible_moves,
                                                                                  n_likely_boards_per_state)
    return weights, counts, likely_boards