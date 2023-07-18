import math
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import List, Optional, Tuple, Dict, Set

import chess.engine
import chess.polyglot
from lczero.backends import Backend, Weights, GameState

from utils import convert_castling_moves_if_any, extend_if_possible, find_best_move_among_all, \
    current_mover_gives_check, pseudo_legal_moves_with_castling_through_check, get_resulting_move, HashableBoard

NULL_MOVE = chess.Move.null()

# VALUE_MATE = 32768
VALUE_MATE = 10000
TIME_USED_FOR_OPERATION = 5

KING_CAPTURE_SCORE = 10
LEAVE_IN_CHECK_SCORE = 10
CHECK_WITHOUT_CAPTURE_SCORE = 5

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

    def get_best_moves_l0(self, boards: List[chess.Board], n_likely_boards_per_state: Optional[int] = None,
                          n_optimistic_boards_per_state: Optional[int] = None) -> Tuple[
        Dict[str, float], Set[chess.Board], Set[chess.Board]]:
        all_moves = set()
        for board in boards:
            all_moves.update(pseudo_legal_moves_with_castling_through_check(board))
        move_weights, move_counts, likely_boards, optimistic_boards = self.get_move_weights_and_move_counts(boards,
                                                                                                            list(
                                                                                                                all_moves),
                                                                                                            n_likely_boards_per_state,
                                                                                                            n_optimistic_boards_per_state)

        return move_weights, likely_boards, optimistic_boards

    def find_best_move_l0(self, boards: Set[chess.Board], possible_moves: List[chess.Move]) -> Optional[chess.Move]:
        move_weights, move_counts, _, _ = self.get_move_weights_and_move_counts(boards, possible_moves)

        # I am FED UP WITH THE ATTACKER BOTS, SO MANUAL CHECKS...
        self.logger.info("Turn: ", self.game_information_db.turn)
        if self.game_information_db.turn == 2 and (
                HashableBoard("rnbqkbnr/ppp2ppp/3N4/3pp3/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 3") in boards or HashableBoard(
                "r1bqkbnr/pppp1ppp/2nN4/4p3/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 3") in boards):
            best_move = chess.Move.from_uci("f8d6")
            return best_move
        elif self.game_information_db.turn == 3 and HashableBoard(
                "r1bqkbnr/pppppppp/8/8/2PP4/2Nn4/PP2PPPP/R1BQKBNR w KQkq - 3 4") in boards:
            best_move = chess.Move.from_uci("d1d3")
            return best_move
        else:
            # Find the best move among all moves
            best_move = find_best_move_among_all(move_weights, move_counts, boards)

        # If we are mated everywhere, return None
        if best_move is None:
            return None

        # Extend the best move if it is possible
        best_move = extend_if_possible(best_move, move_weights, move_counts, boards[0])

        self.logger.info(int(boards[:20]))

        # print(move_weights)
        self.logger.info((sorted(move_weights.items(), key=lambda item: item[1], reverse=True)))
        

        # Convert the castling moves if the best move is a castling move
        best_move = convert_castling_moves_if_any(best_move)

        return best_move

    def get_move_weights_and_move_counts(self, boards: List[chess.Board], possible_moves: List[chess.Move],
                                         n_likely_boards_per_state: Optional[int] = None,
                                         n_optimistic_boards_per_state: Optional[int] = None) -> Tuple[
        Dict, Dict, Set[chess.Board], Set[chess.Board]]:
        move_weights = defaultdict(float)
        move_counts = defaultdict(int)
        all_likely_boards = set()
        all_optimistic_boards = set()

        # Parralel
        with Pool() as p:
            results = p.map(worker, [(board, possible_moves, n_likely_boards_per_state,
                                      n_optimistic_boards_per_state) for board in boards])

        # # Iterative for debugging
        # results = []
        # for board in boards:
        #     results.append(worker((board, possible_moves, n_likely_boards_per_state,
        #                            n_optimistic_boards_per_state)))

        for weights, counts, likely_boards, optimistic_boards in results:
            if likely_boards:
                all_likely_boards.update(likely_boards)

            if optimistic_boards:
                all_optimistic_boards.update(optimistic_boards)

            for move, weight in weights.items():
                move_weights[move] += weight

            for move, count in counts.items():
                move_counts[move] += count

        return move_weights, move_counts, all_likely_boards, all_optimistic_boards

    @staticmethod
    def calculate_move_weights_and_get_likely_boards(board: chess.Board, b, possible_moves: List[chess.Move],
                                                     n_likely_boards_to_return: Optional[int] = None,
                                                     n_optimistic_boards_per_state: Optional[int] = None) -> Tuple[
        Dict, Dict, Optional[Set[chess.Board]], Optional[Set[chess.Board]]]:
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
            return move_weights, move_counts, None, None

        # Second case:
        if board.is_checkmate():
            return move_weights, move_counts, None, None

        game_state = GameState(board.fen())
        i2 = game_state.as_input(b)
        eval = b.evaluate(i2)[0]

        #q = eval.q()  # q will be in [-1, 1]
        # cp = lc0_q_value_to_centipawn_score(q)

        moves = game_state.moves()
        policy_indices = game_state.policy_indices()
        #move_scores = list(zip(moves, eval.p_raw(*policy_indices)))  # list of (move, probability) tuples
        move_scores_softmax = list(zip(moves, eval.p_softmax(*policy_indices)))

        sorted_moves = sorted(move_scores_softmax, key=lambda x: x[1], reverse=True)

        likely_boards = set()
        optimistic_boards = set()

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

            # TODO: Do same for optimistic, i.e. take the n lowest moves:
            if n_optimistic_boards_per_state and idx >= len(sorted_moves) - n_optimistic_boards_per_state:
                optimistic_board = board.copy()
                optimistic_board.push(chess.Move.from_uci(move))
                optimistic_boards.add(optimistic_board)

        board_is_in_check = board.is_check()

        # If we are into check on this board, subtract points for all moves not suggested here
        for possible_move in possible_moves:
            if possible_move.uci() not in move_weights:
                resulting_move = get_resulting_move(board, possible_move)
                if board_is_in_check:
                    if resulting_move in move_weights:
                        move_weights[possible_move.uci()] += move_weights[resulting_move.uci()]
                        move_counts[possible_move.uci()] += 1
                        continue
                    move_weights[possible_move.uci()] -= LEAVE_IN_CHECK_SCORE
                    move_counts[possible_move.uci()] += 1
                    continue


                from_square = possible_move.from_square
                if not board.piece_at(from_square):
                    # Only happens when calculating moves for the opponent
                    move_weights[possible_move.uci()] -= math.log(len(possible_moves) // 2 + 1) * 2.5
                    move_counts[possible_move.uci()] += 1
                    continue

                # What if the move BRINGS us into check?
                if board.piece_at(from_square).piece_type == chess.KING:
                    move_weights[possible_move.uci()] -= LEAVE_IN_CHECK_SCORE
                    move_counts[possible_move.uci()] += 1
                    continue

                if not resulting_move:
                    # Must be pawn move?
                    move_weights[possible_move.uci()] -= math.log(len(possible_moves) // 2 + 1) * 2.5
                    move_counts[possible_move.uci()] += 1
                    continue

                if resulting_move.uci() in move_weights:
                    move_weights[possible_move.uci()] += move_weights[resulting_move.uci()]
                    move_counts[possible_move.uci()] += 1

                else:
                    move_weights[possible_move.uci()] -= math.log(len(possible_moves) // 2 + 1) * 2.5
                    move_counts[possible_move.uci()] += 1

        return move_weights, move_counts, likely_boards, optimistic_boards


def worker(args):
    board, possible_moves, n_likely_boards_per_state, n_optimistic_boards_per_state = args
    weights, counts, likely_boards, optimistic_boards = MoveStrategy.calculate_move_weights_and_get_likely_boards(board,
                                                                                                                  L0_BACKEND,
                                                                                                                  possible_moves,
                                                                                                                  n_likely_boards_per_state,
                                                                                                                  n_optimistic_boards_per_state)
    return weights, counts, likely_boards, optimistic_boards
