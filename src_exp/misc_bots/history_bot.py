import json
from typing import Optional, List, Tuple, Dict

import chess
import chess.polyglot
from chess import Color, Square
from reconchess import Player, WinReason, GameHistory
from loguru import logger as history_logger

from src.boards_tracker import BoardsTracker
from src.scorca.game_information_db import GameInformationDB
from src.scorca.sense_strategy import SenseStrategy, NaiveEntropySense
from src.utils import get_best_moves_l0, get_best_center_from_best_op_moves_dict

chess.Board.__hash__ = lambda self: chess.polyglot.zobrist_hash(self)

class HistoryBot(Player):

    def __init__(self, history_path: str, color: Color, results_path: str):
        self.history = None
        self.logger = history_logger
        self.color = color
        self.color_key = 'true' if self.color else 'false'
        self.load_history(history_path)
        self.len_moves = len(self.history['requested_moves'][self.color_key])
        self.senses = iter(self.history['senses'][self.color_key])
        self.moves = iter(self.history['requested_moves'][self.color_key])

        self.likely_states = set()
        self.opp_move_weights = None
        self.background_calc_finished = True
        self.percentage_calculated = 0
        self.best_moves_for_opponent = None
        self.stop_background_calculation = False

        self.sense_strategy = NaiveEntropySense()


        self.move_count = 0

        self.boards_tracker = BoardsTracker()

        self.results_path = results_path
        self.state_counts = []

        self.game_information_db = GameInformationDB(None, None)
        self.game_information_db.color = color
        self.game_information_db.opponent_color = not color



    def load_history(self, history_path: str):
        with open(history_path, 'r') as f:
            self.history = json.load(f)

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.logger.info('Game starting...')

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.stop_background_calculation = True
        self.logger.critical('*************************')
        self.logger.critical('Handle opponent move result')
        # Wait for background calculation to finish up
        while not self.background_calc_finished:
            pass

        if (self.color and GameInformationDB.turn == 0):
            return

        self.boards_tracker.handle_opponent_move_result(captured_my_piece, capture_square, 300)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        """Choose sense based on history"""
        if self._should_skip_sense():
            return None

        # Likely states sense
        self.logger.critical('SENSING BASED ON LIKELY STATES')
        square:int = self.sense_strategy.sense(self.likely_states, [], game_information_db=self.game_information_db)

        self.logger.critical(f'Square to sense in uci format: {chess.SQUARE_NAMES[square]}')

        # # Opp move weights sense
        # square = get_best_center_from_best_op_moves_dict(self.opp_move_weights)

        return square





    def _should_skip_sense(self):
        if len(self.boards_tracker.possible_states) == 1:
            self.logger.info('Only one possible board, no need to sense')
            return True
        self.logger.info(f"Amount of possible boards: {len(self.boards_tracker.possible_states)}")
        return False

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        pre_sense_state_count = len(self.boards_tracker.possible_states)

        self.boards_tracker.handle_sense_result(sense_result, 300)

        # Store the number of possible states after the sense
        post_sense_state_count = len(self.boards_tracker.possible_states)

        # Add both counts to our list
        self.state_counts.append((pre_sense_state_count, post_sense_state_count))

    def choose_move(self, move_actions: List[chess.Move], seconds_left: int) -> Optional[chess.Move]:
        """Choose move based on history"""
        if self.color and self.move_count%5 == 0:
            print(f"Move {self.move_count} / {self.len_moves}")
        move = chess.Move.from_uci(next(self.moves)['value'])
        self.move_count += 1
        return move

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        self.boards_tracker.handle_own_move_result(requested_move=requested_move or chess.Move.null(),
                                                   taken_move=taken_move or chess.Move.null(),
                                                   captured_opponent_piece=captured_opponent_piece,
                                                   capture_square=capture_square,
                                                   seconds_left=300)
        GameInformationDB.turn += 1

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

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        self.logger.info('Game end')
        self.write_state_counts_to_file()

    def write_state_counts_to_file(self):
        # Construct a dictionary with the sense strategy and state counts
        data = {
            'sense_strategy': 'Likely states sense',
            'state_counts': self.state_counts,
        }

        # Write the data to a JSON file
        with open(f'{self.results_path}', 'w') as f:
            json.dump(data, f)
