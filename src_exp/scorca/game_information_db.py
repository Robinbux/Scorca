from dataclasses import dataclass
from enum import Enum

import chess

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