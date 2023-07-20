import collections
import time
from typing import List, Tuple, Optional, Dict, Set
from itertools import chain

import chess
import chess.polyglot
from joblib import Parallel, delayed

from board_tracker_utils import legal_board_for_sense_result, legal_board_for_own_move_result, \
    next_possible_board_states_based_on_opponent_move_result
from multiprocessing import Pool

from utils import convert_castling_moves_if_any, possible_piece_types_from_move, HashableBoard


class BoardsTracker:

    def __init__(self, logger):
        self.logger = logger
        self.possible_states = {HashableBoard()}
        self.likely_states = {}
        self.transposition_table = {}
        self.taken_moves = []
        self.num_cpus = 4
        self.best_squares_for_opp_to_move_to = [] # We only do that if there are not a lot of opponents boards

    def calculate_likely_states(self, opp_move_weights: Dict[chess.Move, float]) -> None:
        self.likely_states = set()
        copy_opp_move_weights = opp_move_weights.copy()
        # Sort the dictionary by value in ascending order
        sorted_opp_move_weights = sorted(copy_opp_move_weights.items(), key=lambda item: item[1], reverse=True)
        # Determine the cutoff point (half of the dictionary's size)
        cutoff_index = int(len(sorted_opp_move_weights) * 0.2)
        # Remove the bottom 50% of moves
        copy_opp_move_weights = dict(sorted_opp_move_weights[:cutoff_index])

        to_squares_with_possible_piece_types: Dict[chess.Square, Set[chess.PieceType]] = dict()
        moves = list(copy_opp_move_weights.keys())
        moves = [convert_castling_moves_if_any(move) for move in moves]
        for move in moves:
            to_square = move.to_square
            possible_piece_types = possible_piece_types_from_move(move)
            if to_square not in to_squares_with_possible_piece_types:
                to_squares_with_possible_piece_types[to_square] = set(possible_piece_types)
            else:
                to_squares_with_possible_piece_types[to_square].update(set(possible_piece_types))

        possible_states: Set[chess.Board] = self.possible_states
        # Sort out all boards where the to square does not contain at least one of the possible piece types
        for board in possible_states:
            for to_square, possible_piece_types in to_squares_with_possible_piece_types.items():
                piece = board.piece_at(to_square)
                if piece is not None and piece.piece_type in possible_piece_types:
                    self.likely_states.add(board)
                    break

    def handle_sense_result(self, sense_result: List[Tuple[chess.Square, Optional[chess.Piece]]], seconds_left: float = None) -> None:
        start_time = time.time()
        new_boards = []
        for board in self.possible_states:
            if time.time() - start_time >= seconds_left:
                print('ABORTING')
                print(f'Passed time: {time.time() - start_time}')
                print(f'Seconds left: {seconds_left}')
                break  # stops the computation if the time limit has been reached
            new_board = delayed(legal_board_for_sense_result)((board, sense_result))
            new_boards.append(new_board)
        self.possible_states = {board for board in Parallel(n_jobs=self.num_cpus)(new_boards) if board is not None}

    def handle_own_move_result(self,
                               captured_opponent_piece: bool,
                               capture_square: Optional[chess.Square],
                               requested_move: Optional[chess.Move] = chess.Move.null(),
                               taken_move: Optional[chess.Move] = chess.Move.null(),
                               seconds_left: float = None
                               ) -> None:
        # Time roughly 10.000 boards: 1 second
        self.taken_moves.append(taken_move)
        start_time = time.time()
        new_boards = []
        for board in self.possible_states:
            if time.time() - start_time >= seconds_left:
                print('ABORTING')
                print(f'Passed time: {time.time() - start_time}')
                print(f'Seconds left: {seconds_left}')
                break  # stops the computation if the time limit has been reached
            new_board = delayed(legal_board_for_own_move_result)((board, requested_move, taken_move, captured_opponent_piece, capture_square))
            new_boards.append(new_board)
        self.possible_states = {board for board in Parallel(n_jobs=self.num_cpus)(new_boards) if board is not None}

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[chess.Square],
                                    seconds_left: float = None) -> None:
        self.best_squares_for_opp_to_move_to = collections.Counter()

        start_time = time.time()
        results = []
        for board in self.possible_states:
            if time.time() - start_time >= seconds_left:
                self.logger.info('ABORTING')
                self.logger.info(f'Passed time: {time.time() - start_time}')
                self.logger.info(f'Seconds left: {seconds_left}')
                self.possible_states = set(chain.from_iterable(result for result in results))
                break  # stops the computation if the time limit has been reached
            results.append(delayed(next_possible_board_states_based_on_opponent_move_result)
                           ((board, captured_my_piece, capture_square), ))
        self.logger.info('Before parallel operation in handle_opponent_move_result!')
        results = Parallel(n_jobs=self.num_cpus)(results)
        self.possible_states = set(chain.from_iterable(result for result in results))
        self.logger.info('After parallel operation in handle_opponent_move_result!')

