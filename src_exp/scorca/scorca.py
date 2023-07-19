import collections
import json
import logging
import os
import random
import threading
import time
from enum import Enum
from functools import wraps
from typing import Tuple, List, Dict

import chess
import chess.polyglot
import colorlog
from reconchess import Player, Color, Optional, WinReason, GameHistory, Square

from src.boards_tracker import BoardsTracker
from .game_information_db import GameInformationDB
from .move_strategy import MoveStrategy, Engine
from .sense_strategy import NaiveEntropySense, SenseStrategy
from ..utils import  get_best_center_from_best_op_moves_dict, \
    get_move_weights_and_move_counts, find_best_move_among_all, calculate_square_entropy, move_square_away_from_edges, \
    get_best_moves_l0

EXPERIMENT_ENTROPY_DIR = '/Users/robinbux/Desktop/RBC_New/experiments/sense_comp'
PIECE_STATES_REMOVAL_DIR = '/Users/robinbux/Desktop/RBC_New/experiments/piece_states_removal'

LOG_LEVELS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}


class GameStage(Enum):
    OPENING = 'opening'
    MIDDLE = 'middle'
    END = 'end'


def log_states_change(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        pre_operation_state_count = len(self.boards_tracker.possible_states)
        result = func(self, *args, **kwargs)
        post_operation_state_count = len(self.boards_tracker.possible_states)
        self._log_states_difference(func.__name__, pre_operation_state_count, post_operation_state_count)
        return result

    return wrapper

chess.Board.__hash__ = lambda self: chess.polyglot.zobrist_hash(self)


class Scorca(Player):
    def __init__(self, engine: Engine = Engine.SUNFISH , disable_logger: bool = False, history_path:str = None, results_path:str = None):
        self.likely_states = set()
        self.opp_move_weights = None
        self.background_calc_finished = True
        self.percentage_calculated = 0
        self.best_moves_for_opponent = None
        self.stop_background_calculation = False
        self.engine = engine
        self._initialize_game_state()
        self._configure_logger(disable_logger)
        self._initialize_strategy(engine)
        self.seconds_left = 100000
        self.best_opponents_move = []

        self.load_history(history_path)


        self.results_path = results_path


    def load_history(self, history_path: str):
        with open(history_path, 'r') as f:
            self.history = json.load(f)

    def reset(self):
        """
        Reset the state of the object to a fresh initialized state.
        """
        self._initialize_game_state()
        self.move_strategy.use_opening_book = True
        self.move_strategy.initialize_stockfish_engine()
        self.seconds_left = 100000
        self.best_opponents_move = []

    def _initialize_game_state(self):
        self.state_counts = []
        self.color = None
        self.board = None
        self.out_of_time = False
        self.game_information_db = GameInformationDB(None, None)
        self.boards_tracker = BoardsTracker()

    def _configure_logger(self, disable_logger: bool):
        self.logger = logging.getLogger(f'scorca_{id(self)}')
        if disable_logger:
            self.logger.setLevel(logging.CRITICAL + 1)
        else:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setFormatter(self._get_log_formatter())
            self.logger.addHandler(handler)

    @staticmethod
    def _get_log_formatter():
        return colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(asctime)s [%(blue)s%(name)s%(reset)s] "
            "%(message_log_color)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'message': {
                    'CRITICAL': 'bold',
                },
            },
            style='%'
        )

    def _initialize_strategy(self, engine: Engine):
        self.move_strategy = MoveStrategy(self.game_information_db, self.logger, engine)
        self.sense_strategy_name = 'naive_entropy_sense'
        self.sense_strategy = NaiveEntropySense(self.move_strategy)
        self.ponders = []
        self.ponders = []

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.logger.info('Game starting...')
        self.color = color
        self.board = board
        self.game_information_db.color = color
        self.game_information_db.opponent_color = not color

        self.color_key = 'true' if self.color else 'false'

        self.len_moves = len(self.history['requested_moves'][self.color_key])
        self.senses = iter(self.history['senses'][self.color_key])
        self.moves = iter(self.history['requested_moves'][self.color_key])

    @log_states_change
    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.stop_background_calculation = True
        self.logger.critical('*************************')
        self.logger.critical('Handle opponent move result')
        # Wait for background calculation to finish up
        while not self.background_calc_finished:
            pass

        if (self.color and self.game_information_db.turn == 0) or self.out_of_time:
            return
        start_time = time.time()
        self.boards_tracker.handle_opponent_move_result(captured_my_piece=captured_my_piece,
                                                        capture_square=capture_square,
                                                        seconds_left=self.seconds_left,
                                                        stockfish_engine=self.move_strategy.stockfish_engine)

        if len(self.boards_tracker.possible_states) > 1 and self.opp_move_weights:
            if capture_square is not None:
                # We remove all moves that don't go to the capture square
                self.opp_move_weights = {key: value for key, value in self.opp_move_weights.items() if key.to_square == capture_square}

        self.logger.info(f'Amount of likely states before: {len(self.likely_states)}')
        # Remove likely states that are not possible anymore
        self.likely_states &= self.boards_tracker.possible_states
        self.logger.info(f'Amount of likely states after: {len(self.likely_states)}')

        self.logger.info(f"Time to handle opponent move result: {time.time() - start_time}")

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        if self._should_skip_sense():
            return None
        if self._is_time_low(seconds_left):
            return self._choose_random_action(sense_actions)

        self.seconds_left = seconds_left
        return self._perform_and_log_sense_action()

    def _should_skip_sense(self):
        if len(self.boards_tracker.possible_states) == 1:
            self.logger.info('Only one possible board, no need to sense')
            return True
        self.logger.info(f"Amount of possible boards: {len(self.boards_tracker.possible_states)}")
        return False

    def _is_time_low(self, seconds_left: float):
        self.logger.info(f"Seconds left: {seconds_left}")
        if seconds_left < 4 or self.out_of_time:
            self.logger.critical('Time low, doing random sense')
            self.out_of_time = True
            return True
        return False

    def _choose_random_action(self, actions: List):
        return random.choice(actions)

    def _perform_and_log_sense_action(self):
        start_time = time.time()
        # if self.percentage_calculated > 0.4 and self.best_moves_for_opponent:
        #     square = get_best_center_from_best_op_moves_dict(self.best_moves_for_opponent)
        #     self.logger.info("Used stockfish to find square!")\

        # if len(self.likely_states) > 1:
        #     self.logger.critical('SENSING BASED ON LIKELY STATES')
        #     square = self.sense_strategy.sense(self.likely_states, self.ponders,
        #                                        game_information_db=self.game_information_db)

        if self.opp_move_weights:
            self.logger.critical('SENSING BASED ON OPP MOVE WEIGHTS')
            #test = get_best_center_from_best_op_moves_dict(self.opp_move_weights)
            square = get_best_center_from_best_op_moves_dict(self.opp_move_weights)
            self.logger.info("Used stockfish to find square!")

        else:
            self.logger.critical('SENSING BASED ON POSSIBLE STATES')
            square = self.sense_strategy.sense(self.boards_tracker.possible_states, self.ponders, game_information_db=self.game_information_db)

        # move_values_dict, move_count_dict = get_move_weights_and_move_counts(self.boards_tracker.possible_states)
        # best_move = find_best_move_among_all(move_values_dict, move_count_dict, self.boards_tracker.possible_states)
        # # Check entropy of to square
        # square_entropy = calculate_square_entropy(self.boards_tracker.possible_states, best_move.to_square)
        # print("*******************************")
        # print("*******************************")
        # print(f'Entropy: {square_entropy}')
        # print("*******************************")
        # print("*******************************")
        #
        # if square_entropy > 0.7:
        #     print('WE DID ENTROPY SINGLE SQUARE SENSE')
        #     print(f'Best move: {best_move}')
        #     square = move_square_away_from_edges(best_move.to_square)
        # elif self.percentage_calculated > 0.4 and self.best_moves_for_opponent:
        #     square = get_best_center_from_best_op_moves_dict(self.best_moves_for_opponent)
        #     self.logger.info("Used stockfish to find square!")
        # else:
        #     square = self.sense_strategy.sense(self.boards_tracker.possible_states, self.ponders, game_information_db=self.game_information_db)

        self.logger.info(f"Time used for sense: {time.time() - start_time:.4f} seconds")
        self.logger.debug(f"Sensed square: {chess.SQUARE_NAMES[square] if square else 'None'}")
        return square

    @log_states_change
    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        if self.out_of_time:
            return
        states_before = len(self.boards_tracker.possible_states)
        self.boards_tracker.handle_sense_result(sense_result, seconds_left=self.seconds_left)
        states_after = len(self.boards_tracker.possible_states)
        self.state_counts.append((states_before, states_after))


        # Update likely states again
        self.logger.info(f'Amount of likely states before: {len(self.likely_states)}')
        # Remove likely states that are not possible anymore
        self.likely_states &= self.boards_tracker.possible_states
        self.logger.info(f'Amount of likely states after: {len(self.likely_states)}')


    def choose_move(self, move_actions: List[chess.Move], seconds_left: int) -> Optional[chess.Move]:
        self.logger.info(f"Seconds MOVE: {seconds_left}")

        self.boards_tracker.likely_states = set()
        # if len(self.boards_tracker.possible_states) > 1:
        #     self.boards_tracker.calculate_likely_states(self.opp_move_weights)
        #     self.logger.critical(f"Amount of possible states: {len(self.boards_tracker.possible_states)}")
        #     self.logger.critical(f'Amount of likely states: {len(self.boards_tracker.likely_states)}')

        # self.logger.debug(f"Possible moves: {move_actions}")
        self.seconds_left = seconds_left
        # if self._is_time_low(seconds_left):
        #     return self._choose_random_action(move_actions)
        # return self._perform_and_log_move_action(move_actions, seconds_left)
        next_move = next(self.moves)
        if not next_move:
            return None
        print(f'Next move: {next_move["value"]}')
        move = chess.Move.from_uci(next_move['value'])
        return move

    def _perform_and_log_move_action(self, move_actions: List[chess.Move], seconds_left: int):
        start_time = time.time()
        states_to_use = self.likely_states if len(self.likely_states) > 0 else self.boards_tracker.possible_states
        move, ponders = self.move_strategy.move(move_actions, states_to_use, seconds_left)
        # If move is none, pick random move
        if move is None:
            move = random.choice(move_actions)
        self.ponders = ponders
        self.logger.info(f"Time used for move: {time.time() - start_time:.4f} seconds")
        self.logger.debug(f"Requested Move: {move.uci() if move else 'None'}")
        self.seconds_left = seconds_left - (time.time() - start_time)
        return move

    @log_states_change
    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        self.logger.info('Handle move result')
        self.logger.info(f"Seconds left: {self.seconds_left}")
        if self.out_of_time:
            return

        self.boards_tracker.handle_own_move_result(requested_move=requested_move or chess.Move.null(),
                                                   taken_move=taken_move or chess.Move.null(),
                                                   captured_opponent_piece=captured_opponent_piece,
                                                   capture_square=capture_square,
                                                   seconds_left=self.seconds_left)


        # Start background_calculation in a new thread
        self.stop_background_calculation = False
        background_thread = threading.Thread(target=self.background_calculation)
        background_thread.daemon = True  # Ensures that background threads do not prevent program termination
        background_thread.start()

        self.game_information_db.turn += 1
        self.logger.info('Opponents turn starting...\n')

    def background_calculation(self):
        self.background_calc_finished = False
        self.opp_move_weights = {}
        self.likely_states = set()
        self.percentage_calculated = 0
        self.best_moves_for_opponent: Dict[Tuple[chess.Square, bool], float] = {}

        opp_move_weights_str_keys, likely_states = get_best_moves_l0(self.boards_tracker.possible_states, n_likely_boards_per_state=2)
        self.logger.critical(f'{len(likely_states)} likely states vs {len(self.boards_tracker.possible_states)} possible states')
        self.likely_states = likely_states
        # Convert string keys to chess.Move keys...
        #print("OPP MOVE WEIGHTS STR KEYS")
        #print(opp_move_weights_str_keys)
        self.opp_move_weights = {chess.Move.from_uci(key): value for key, value in opp_move_weights_str_keys.items()}

        # Play moves on all possible boards for

        self.logger.info("Stopped background calculation!")
        self.background_calc_finished = True

    # def background_calculation(self):
    #     self.background_calc_finished = False
    #     self.percentage_calculated = 0
    #     self.best_moves_for_opponent: Dict[Tuple[chess.Square, bool], float] = {}
    #
    #     possible_boards = self.boards_tracker.possible_states
    #     n_best_moves = 3
    #
    #     amount_of_boards_calculated = 0
    #     # Iterate over all boards in random order
    #     for board in possible_boards:
    #         # Check if should stop background calculation
    #         if self.stop_background_calculation:
    #             break
    #
    #         # Process the board
    #         best_moves = get_best_n_moves(board, n=n_best_moves, seconds=0.1,
    #                                       stockfish_engine=self.move_strategy.stockfish_engine)
    #
    #         # Accumulate the scores in the dictionary
    #         for key, score in best_moves.items():
    #             if key in self.best_moves_for_opponent:
    #                 self.best_moves_for_opponent[key] += 1
    #             else:
    #                 self.best_moves_for_opponent[key] = 1
    #         amount_of_boards_calculated += 1
    #
    #     self.percentage_calculated = amount_of_boards_calculated / len(possible_boards)
    #     self.logger.info(f"Stopped background calculation at {self.percentage_calculated:.2%}")
    #     self.background_calc_finished = True


    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        self.logger.info('Game end')
        self._write_state_counts_to_file()

    def _write_state_counts_to_file(self):
        data = {
            'sense_strategy':'likely_states',
            'state_counts': self.state_counts,
        }
        new_dir_index = len(os.listdir(EXPERIMENT_ENTROPY_DIR))
        new_dir_name = 'game_{:03d}'.format(new_dir_index)
        with open(self.results_path, 'w') as f:
            json.dump(data, f)

    def _log_states_change(self, operation: str, func, *args):
        pre_operation_state_count = len(self.boards_tracker.possible_states)
        func(*args)
        post_operation_state_count = len(self.boards_tracker.possible_states)
        self._log_states_difference(operation, pre_operation_state_count, post_operation_state_count)

    def _log_states_difference(self, operation: str, before: int, after: int):
        if before <= 0  or after < 0:
            return
        self.logger.critical(f"States changes {operation}: {before} -> {after}")
        percentage_change = (after - before) / before * 100
        self.logger.critical(f"{after - before} states {'added' if after > before else 'removed'}! "
                             f"({percentage_change:.2f}%)")
