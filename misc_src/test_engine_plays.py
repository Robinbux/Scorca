import chess
import chess.engine
import chess.polyglot
import time

from misc_src.load_chess_boards import load_chess_boards
from src.utils import get_stockfish_board_score


class Test:

    def __init__(self):
        self.stockfish_engine = None
        self._initialize_stockfish_engine()

    def _initialize_stockfish_engine(self):
        self.stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, setpgrp=True)
        self.stockfish_engine.configure({
            "Use NNUE": True,
            'Hash': 2048,
            'Threads': 5,
        })

    def quit_engine(self):
        self.stockfish_engine.quit()

    def test_analyze(self, board, analyze_time):
        result = self.stockfish_engine.analyse(board, chess.engine.Limit(time=analyze_time),
                                                    info=chess.engine.INFO_SCORE)
        return result

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

boards = load_chess_boards()
test = Test()

board_index = 24

print(boards[board_index].fen())

times = [0.001, 0.01, 0.05, 0.1, 1, 3, 6, 10]

board_to_use = chess.Board('5k2/8/8/8/7r/6r1/8/K7 b - - 0 1')

# for analyze_time in times:
#     result = test.test_analyze(board_to_use, analyze_time=analyze_time)
#     score = result['score'].pov(chess.WHITE)
#     print(f'Analyze time: {analyze_time} seconds')
#     print(f'Score: {score}')
#     print(f'Nodes: {result["nodes"]}')
#     print('------------------')
chess.Board.__hash__ = lambda self: chess.polyglot.zobrist_hash(self)

result = get_stockfish_board_score(board_to_use, chess.WHITE, test.stockfish_engine)

print(result)