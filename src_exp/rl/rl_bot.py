from reconchess import *
import numpy as np
from enum import Enum
from src.boards_tracker import BoardsTracker
from dataclasses import dataclass
from loguru import logger as rl_logger

# Define constants for rewards
REWARD_WIN = 100
REWARD_LOSE = -100
REWARD_DRAW = 10

# Define constants for piece rewards
REWARD_PAWN = 100
REWARD_KNIGHT = 300
REWARD_BISHOP = 300
REWARD_ROOK = 500
REWARD_QUEEN = 900
REWARD_KING = 10000
REWARD_NONE = 0

# Define game stages as an Enum
class GameStage(Enum):
    OPENING = 'opening'
    MIDDLE = 'middle'
    END = 'end'

@dataclass
class GameInformationDB:
    color: chess.Color
    opponent_color: chess.Color
    turn: int = 0
    game_stage: GameStage = GameStage.OPENING

class RLBot(Player):

    def __init__(self, rl_agent):
        self.rl_agent = rl_agent
        self.color = None
        self.game_information_db = None
        self.boards_tracker = BoardsTracker(self.game_information_db)
        self.logger = rl_logger

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.color = color
        self.rl_agent.reset()
        self.game_information_db = GameInformationDB(color, not color)
        self.logger.info(f'Game started. Player color: {self.color}, Opponent: {opponent_name}')

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        if self.color and self.game_information_db.turn == 0:
            return

        self.boards_tracker.handle_opponent_move_result(captured_my_piece, capture_square)

        if captured_my_piece:
            piece_type = self.boards_tracker.get_piece_type_at_square(capture_square)
            reward = -self.get_reward_for_piece_type(piece_type)  # Negative reward for losing a piece
            self.rl_agent.remember(self.get_state(), None, reward, self.get_state(), done=False)
            self.logger.info(f'Opponent captured a piece at square: {capture_square}')

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
    Optional[Square]:
        if len(self.boards_tracker.possible_boards) == 1:
            self.logger.info('Only one possible board, no need to sense')
            return None
        state = self.get_state()  # Convert the current game state to a format that can be input to the RL agent
        action = self.rl_agent.choose_sense(state)  # Let the RL agent choose the sense action
        self.logger.info(f'Chosen sense action: {action}')
        return action

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # Update the state of the game based on the sense result
        self.boards_tracker.handle_sense_result(sense_result)
        self.logger.info(f'Sense result: {sense_result}')

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        state = self.get_state()  # Convert the current game state to a format that can be input to the RL agent
        action = self.rl_agent.choose_move(state, move_actions)  # Let the RL agent choose the move action
        self.logger.info(f'Chosen move action: {action}')
        return action

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        state_before = self.get_state()
        self.boards_tracker.handle_own_move_result(requested_move=requested_move or chess.Move.null(),
                                                   taken_move=taken_move or chess.Move.null(),
                                                   captured_opponent_piece=captured_opponent_piece,
                                                   capture_square=capture_square)
        self.game_information_db.turn += 1

        if captured_opponent_piece:
            piece_type = self.boards_tracker.get_piece_type_at_square(capture_square)
            reward = self.get_reward_for_piece_type(piece_type)
            self.rl_agent.remember(state_before, taken_move, reward, self.get_state(), done=False)
            self.logger.info(
                f'Requested move: {requested_move}, Taken move: {taken_move}, Captured opponent piece: {captured_opponent_piece}')

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        if winner_color == self.color:
            reward = REWARD_WIN  # Large reward for winning
        elif winner_color is not self.color:
            reward = REWARD_LOSE  # Large negative reward for losing
        else:
            reward = REWARD_DRAW  # Small reward for draw

        self.rl_agent.remember(self.get_state(), None, reward, self.get_state(), done=True)

    def get_state(self):
        state_representation = []
        # Flatten each board into a 64-length list and add it to our state representation
        for board in self.boards_tracker.possible_boards:
            flat_board = np.zeros((8, 8, 13))  # Start with an empty board

            for i in range(64):
                piece = board.piece_at(i)
                if piece is not None:
                    flat_board[i // 8, i % 8, (piece.piece_type - 1) * 2 + int(piece.color)] = 1
                else:
                    flat_board[i // 8, i % 8, 12] = 1  # Set the last channel to 1 for empty squares

            state_representation.append(flat_board)

        state_representation = np.array(state_representation)
        return state_representation

    def get_reward_for_piece_type(self, piece_type):
        if piece_type == chess.PAWN:
            return REWARD_PAWN
        elif piece_type == chess.KNIGHT:
            return REWARD_KNIGHT
        elif piece_type == chess.BISHOP:
            return REWARD_BISHOP
        elif piece_type == chess.ROOK:
            return REWARD_ROOK
        elif piece_type == chess.QUEEN:
            return REWARD_QUEEN
        elif piece_type == chess.KING:
            return REWARD_KING
        else:
            return REWARD_NONE
