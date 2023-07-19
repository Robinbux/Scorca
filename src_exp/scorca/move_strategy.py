import random
import scipy
from scipy.stats import beta
import time as t
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from itertools import zip_longest
from typing import List, Optional, Tuple, Dict, Set, Any


import chess.engine
from chess import Move
from chess.polyglot import open_reader

from src.boards_tracker.board_tracker_utils import pseudo_legal_moves_with_castling_through_check
from src.boards_tracker.boards_tracker import BoardsTracker
from src.rbc_sunfish import *
from src.utils import current_mover_gives_check, get_resulting_move, \
    convert_centipawn_score_to_win_probability, find_best_move_l0

OPENING_BOOK_PATH = '../opening_book/Human.bin'
STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'
NNUE_PATH = "/Users/robinbux/Desktop/RBC_New/NNUE/nn-e1fb1ade4432.nnue"

ESTIMATED_MOVES_TO_PLAY = 6

NULL_MOVE = chess.Move.null()

from stockfish import Stockfish



@dataclass
class BotParameters:
    mate_score_base: int
    check_exposed_value: int
    king_capture_score: int
    boards_per_best_worst_n: int
    evaluated_boards_amount_limit: int
    remove_moves_percentile: int


class Engine(Enum):
    SUNFISH = 'sunfish'
    STOCKFISH = 'stockfish'


class MoveStrategy:
    def __init__(self, game_information_db, logger, engine=Engine.SUNFISH):
        self.logger = logger
        self.use_opening_book = False
        self.opening_book_reader = open_reader(OPENING_BOOK_PATH)
        self.game_information_db = game_information_db
        self.engine = engine
        self.stockfish_engine = None

        self.bot_parameters = None

    def set_options(self, bot_parameters: BotParameters):
        # Convert certrain params to ints
        bot_parameters.boards_per_best_worst_n = int(bot_parameters.boards_per_best_worst_n)
        bot_parameters.evaluated_boards_amount_limit = int(bot_parameters.evaluated_boards_amount_limit)
        self.bot_parameters = bot_parameters

    def get_options(self):
        return asdict(self.bot_parameters)

    def move(self, move_actions: List[chess.Move], states: Set[chess.Board], seconds_left: int) -> Tuple[Optional[
        chess.Move], List[chess.Move]]:
        self.move_actions = move_actions
        #self.boards_tracker = boards_tracker
        #states_to_use = self.boards_tracker.likely_states if len(self.boards_tracker.likely_states) > 0 else self.boards_tracker.possible_states
        match self.engine:
            case (Engine.STOCKFISH):
                return self._stockfish_move(states, seconds_left)

    def _stockfish_move(self, possible_boards: List[chess.Board], seconds_left: int) -> Tuple[
        Optional[chess.Move], List[chess.Move]]:
        if self.use_opening_book:
            if opening_move := self._get_opening_book_move(possible_boards):
                return opening_move, []

        possible_boards = list(possible_boards)
        best_move = find_best_move_l0(possible_boards, self.move_actions)
        return best_move, []

    def convert_scores_to_win_probability(self, boards_with_scores: List[Tuple[chess.Board, float]]) -> List[
        Tuple[chess.Board, float]]:
        return [(board, convert_centipawn_score_to_win_probability(score)) for board, score in boards_with_scores]

    def limit_boards(self, possible_boards: List[chess.Board], max_boards: int) -> List[chess.Board]:
        # Sample with a uniform distribution
        if len(possible_boards) > max_boards:
            # Create an array of indices that divide the list into equally spaced segments
            indices = np.linspace(0, len(possible_boards), max_boards + 1, dtype=int)
            # Sample one index from each segment
            sampled_indices = [np.random.randint(low=indices[i], high=indices[i + 1]) for i in range(len(indices) - 1)]
            # Use the sampled indices to get the boards
            return [possible_boards[i] for i in sampled_indices]
        return possible_boards

    def calculate_boards_scores(self, possible_boards: List[chess.Board]) -> List[Tuple[chess.Board, float]]:
        boards_with_scores = [
            (board, get_stockfish_board_score(board, self.game_information_db.color, self.stockfish_engine, restart_engine_func=self.initialize_stockfish_engine)) for board
            in possible_boards]
        # Sort out none boards
        return [board_with_score for board_with_score in boards_with_scores if board_with_score[1] is not None]

    def get_bottom_and_top_boards(self, boards_with_scores: List[Tuple[chess.Board, float]],
                                  amount_of_boards_to_evaluate: int) -> Tuple[List, List]:
        boards_len = len(boards_with_scores)
        if boards_len >= amount_of_boards_to_evaluate * 2:
            bottom_n_boards = sorted(boards_with_scores, key=lambda x: x[1])[:amount_of_boards_to_evaluate]
            top_n_boards = sorted(boards_with_scores, key=lambda x: x[1], reverse=True)[:amount_of_boards_to_evaluate]
        else:
            bottom_n_boards = sorted(boards_with_scores, key=lambda x: x[1])[:boards_len // 2]
            top_n_boards = sorted(boards_with_scores, key=lambda x: x[1], reverse=True)[boards_len // 2:]
        return bottom_n_boards, top_n_boards

    def process_boards(self, all_boards: List[Tuple[chess.Board, float]], time_limit_seconds: float,
                       possible_moves: List[chess.Move]):
        move_values_dict = defaultdict(float)
        move_count_dict = defaultdict(int)
        move_boards_dict = defaultdict(set)
        ucb_dict = defaultdict(float)

        for move in possible_moves:
            move_values_dict[move]
            move_count_dict[move]
            move_boards_dict[move]
            ucb_dict[move]

        num_boards_to_sample = 2
        total_iterations = 0

        start_time = t.time()

        while t.time() - start_time < time_limit_seconds:
            moves = list(move_values_dict.keys())
            total_iterations += 1

            # Compute the upper confidence bounds for each move
            for move in moves:
                move_count = move_count_dict[move]
                if move_count > 0:
                    # UCB1 formula
                    ucb_dict[move] = move_values_dict[move] + np.sqrt(2 * np.log(total_iterations) / move_count)
                else:
                    ucb_dict[move] = float('inf')

            # Choose the move with the highest upper confidence bound
            sampled_move = max(ucb_dict, key=ucb_dict.get)
            unexplored_boards = [board for board in all_boards if board not in move_boards_dict[sampled_move]]

            # If all boards have been explored for all moves, break the loop
            if all(len(move_boards_dict[move]) == len(all_boards) for move in moves):
                break

            if not unexplored_boards:
                continue

            boards_to_explore = random.sample(unexplored_boards, min(num_boards_to_sample, len(unexplored_boards)))

            for board in boards_to_explore:
                move_boards_dict[sampled_move].add(board)
                pseudo_legal_moves = pseudo_legal_moves_with_castling_through_check(board)
                resulting_move = get_resulting_move(board, sampled_move, pseudo_legal_moves)
                if resulting_move is None:
                    continue

                board_copy = board.copy()

                if current_mover_gives_check(board_copy):
                    pre_move_board_value = self.bot_parameters.king_capture_score * 0.5
                else:
                    pre_move_board_value = get_stockfish_board_score(board_copy, self.game_information_db.color,
                                                                     self.stockfish_engine,
                                                                     restart_engine_func=self.initialize_stockfish_engine)
                    if not pre_move_board_value:
                        continue

                move_count_dict[sampled_move] += 1

                op_king_square = board_copy.king(not board_copy.turn)
                if resulting_move.to_square == op_king_square:
                    move_diff = self.bot_parameters.king_capture_score - pre_move_board_value
                elif board_copy.is_into_check(resulting_move):
                    move_diff = self.bot_parameters.check_exposed_value - pre_move_board_value
                else:
                    board_copy.push(resulting_move)
                    board_copy.clear_stack()
                    post_move_board_value = get_stockfish_board_score(board_copy, self.game_information_db.color,
                                                                      self.stockfish_engine,
                                                                      restart_engine_func=self.initialize_stockfish_engine)
                    move_diff = post_move_board_value - pre_move_board_value if post_move_board_value is not None else 0

                # Update the move value with the incremental average
                move_values_dict[sampled_move] += (move_diff - move_values_dict[sampled_move]) / move_count_dict[
                    sampled_move]

        best_move = max(move_values_dict, key=move_values_dict.get)
        self.logger.debug(f"Turn: {self.game_information_db.turn}")
        self.logger.debug(move_values_dict)
        return move_values_dict, move_count_dict, {}



    # def process_boards(self, all_boards: List[Tuple[chess.Board, float]], time_limit_seconds: float,
    #                    possible_moves: List[chess.Move]):
    #     move_values_dict = defaultdict(float)
    #     move_count_dict = defaultdict(int)
    #     move_boards_dict = defaultdict(set)
    #
    #     for move in possible_moves:
    #         move_values_dict[move]
    #         move_count_dict[move]
    #         move_boards_dict[move]
    #
    #     temperature = 1.0  # Can be tuned
    #     num_boards_to_sample = 2
    #
    #     start_time = t.time()
    #
    #     while t.time() - start_time < time_limit_seconds:
    #         moves = list(move_values_dict.keys())
    #         move_values = list(move_values_dict.values())
    #         probabilities = compute_softmax_probabilities(move_values, temperature)
    #
    #         sampled_move = sample_action(moves, probabilities)
    #
    #         unexplored_boards = [board for board in all_boards if board not in move_boards_dict[sampled_move]]
    #
    #         # If all boards have been explored for all moves, break the loop
    #         if all(len(move_boards_dict[move]) == len(all_boards) for move in moves):
    #             break
    #
    #         if not unexplored_boards:
    #             continue
    #
    #         boards_to_explore = random.sample(unexplored_boards, min(num_boards_to_sample, len(unexplored_boards)))
    #
    #         for board, score in boards_to_explore:
    #             move_boards_dict[sampled_move].add(board)
    #             pseudo_legal_moves = pseudo_legal_moves_with_castling_through_check(board)
    #             resulting_move = get_resulting_move(board, sampled_move, pseudo_legal_moves)
    #             if resulting_move is None:
    #                 continue
    #
    #             board_copy = board.copy()
    #
    #             if current_mover_gives_check(board_copy):
    #                 pre_move_board_value = self.bot_parameters.king_capture_score * 0.5
    #             else:
    #                 pre_move_board_value = get_stockfish_board_score(board_copy, self.game_information_db.color,
    #                                                                  self.stockfish_engine,
    #                                                                  restart_engine_func=self.initialize_stockfish_engine)
    #                 if not pre_move_board_value:
    #                     continue
    #
    #             move_count_dict[sampled_move] += 1
    #
    #             op_king_square = board_copy.king(not board_copy.turn)
    #             if resulting_move.to_square == op_king_square:
    #                 move_diff = self.bot_parameters.king_capture_score - pre_move_board_value
    #             elif board_copy.is_into_check(resulting_move):
    #                 move_diff = self.bot_parameters.check_exposed_value - pre_move_board_value
    #             else:
    #                 board_copy.push(resulting_move)
    #                 board_copy.clear_stack()
    #                 post_move_board_value = get_stockfish_board_score(board_copy, self.game_information_db.color,
    #                                                                   self.stockfish_engine,
    #                                                                   restart_engine_func=self.initialize_stockfish_engine)
    #                 move_diff = post_move_board_value - pre_move_board_value if post_move_board_value is not None else 0
    #
    #             # Update the move value with the incremental average
    #             move_values_dict[sampled_move] += (move_diff - move_values_dict[sampled_move]) / move_count_dict[
    #                 sampled_move]
    #
    #     best_move = max(move_values_dict, key=move_values_dict.get)
    #     self.logger.debug(f"Turn: {self.game_information_db.turn}")
    #     self.logger.debug(move_values_dict)
    #     return move_values_dict, move_count_dict, {}

    # def process_boards(self, all_boards: List[Tuple[chess.Board, float]], time_limit_seconds: float, possible_moves: List[chess.Move]) -> tuple[
    #     defaultdict[Any, float] | defaultdict[str, float], defaultdict[Any, int] | defaultdict[str, int], set[Move]]:
    #     move_values_dict = defaultdict(float)
    #     move_count_dict = defaultdict(int)
    #     moves_that_could_kill_us = set()
    #
    #     start_time = t.time()
    #
    #     # Process each board
    #     for board, score in all_boards:
    #         if t.time() - start_time > time_limit_seconds:
    #             self.logger.debug("Time limit reached, stopping processing boards.")
    #             break
    #
    #         board_copy = board.copy()
    #
    #         #multiplication_score = abs(score - 0.5) * 2
    #         multiplication_score = 1
    #
    #         if current_mover_gives_check(board_copy):
    #             pre_move_board_value = self.bot_parameters.king_capture_score * 0.5
    #         else:
    #             pre_move_board_value = get_stockfish_board_score(board, self.game_information_db.color,
    #                                                              self.stockfish_engine, restart_engine_func=self.initialize_stockfish_engine)
    #             if not pre_move_board_value:
    #                 continue
    #
    #         pseudo_legal_moves = pseudo_legal_moves_with_castling_through_check(board_copy)
    #         op_king_square = board_copy.king(not board_copy.turn)
    #
    #         # Process each move
    #         for move in self.move_actions:
    #             resulting_move = get_resulting_move(board_copy, move, pseudo_legal_moves)
    #             if resulting_move is None:
    #                 continue
    #
    #             move_count_dict[move] += 1
    #
    #             # Check if move 'captures' the opponents king
    #             if resulting_move.to_square == op_king_square:
    #                 move_values_dict[move] += self.bot_parameters.king_capture_score * multiplication_score
    #                 continue
    #
    #             # Check if move puts us into check
    #             if board_copy.is_into_check(resulting_move):
    #                 move_values_dict[move] += self.bot_parameters.check_exposed_value * multiplication_score
    #                 moves_that_could_kill_us.add(move)
    #                 continue
    #
    #             board_copy.push(resulting_move)
    #             board_copy.clear_stack()
    #
    #             # Compute post-move board value and difference
    #             post_move_board_value = get_stockfish_board_score(board_copy, self.game_information_db.color,
    #                                                               self.stockfish_engine, restart_engine_func=self.initialize_stockfish_engine)
    #             board_copy = board.copy()
    #
    #             if post_move_board_value is not None:
    #                 move_diff = post_move_board_value - pre_move_board_value
    #                 move_values_dict[move] += move_diff * multiplication_score
    #
    #     return move_values_dict, move_count_dict, moves_that_could_kill_us

    def prune_and_normalize_moves(self, move_values_dict: Dict[chess.Move, float],
                                  move_count_dict: Dict[chess.Move, int]) -> Dict[chess.Move, float]:
        percentile = np.percentile(list(move_count_dict.values()), self.bot_parameters.remove_moves_percentile)
        moves_to_remove = {
            move
            for move, count in move_count_dict.items()
            if count < percentile
        }
        for move in moves_to_remove:
            del move_values_dict[move]
        # Normalizing move values
        min_move_value = min(move_values_dict.values())
        max_move_value = max(move_values_dict.values())
        range_move_value = max_move_value - min_move_value
        if range_move_value == 0:  # handle the case when all values are the same
            for move in move_values_dict:
                move_values_dict[move] = 1
        else:
            for move, score in move_values_dict.items():
                move_values_dict[move] = (score - min_move_value) / range_move_value
        return move_values_dict

    def select_best_move(self, move_values_dict: Dict[chess.Move, float],
                         moves_that_could_kill_us: Set[chess.Move]) -> chess.Move:
        # TODO: Maybe minus for moves that could kill us?
        return max(move_values_dict, key=move_values_dict.get)

    def get_best_stockfish_move(self, board: chess.Board, time: float = 3) -> chess.Move:
        # Check if we can capture the opponent's king
        op_king_square = board.king(not board.turn)
        pseudo_legal_moves = pseudo_legal_moves_with_castling_through_check(board)
        for move in pseudo_legal_moves:
            if move.to_square == op_king_square:
                return move
        board.clear_stack()
        self.logger.info(f"Stockfish analyzing board {board.fen()}")
        move = self.stockfish_engine.play(board, chess.engine.Limit(time=time)).move
        return move

    def _get_opening_book_move(self, possible_boards: List[chess.Board]) -> Optional[chess.Move]:
        moves_counter = self._count_moves_from_opening_book(possible_boards)

        if not moves_counter:
            self.use_opening_book = False
            self.logger.debug('DONE WITH OPENING BOOK')
            return None

        return moves_counter.most_common(1)[0][0]

    def _count_moves_from_opening_book(self, possible_boards):
        moves_counter = Counter()

        for board in possible_boards:
            for entry in self.opening_book_reader.find_all(board):
                moves_counter[entry.move] += entry.weight

        return moves_counter
