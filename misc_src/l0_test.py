import unittest

import chess
from chess import Board
from loguru import logger

from src.scorca.game_information_db import GameInformationDB
from src.scorca.move_strategy import MoveStrategy
from src.scorca.utils import HashableBoard


class TestFindBestMove(unittest.TestCase):

    def test_case_1(self):
        color = chess.BLACK
        game_information_db = GameInformationDB(color, not color)

        move_strategy = MoveStrategy(game_information_db, logger)

        boards = [Board('8/8/2p2k1p/1p3P2/p7/P4K2/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3K2/p5P1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p1Kk1p/1p6/p5P1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p2K3/p5P1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3P2/p7/P5K1/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3P2/p5K1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3P2/p4K2/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2kKp/1p6/p5P1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p6/p3K1P1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p4K1/p5P1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p2KP2/p7/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3P2/p7/P3K3/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3P2/p3K3/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3KP1/p7/P7/1PP5/8 b - - 0 50'),
                  Board('8/8/2p2k1p/1p6/p4KP1/P7/1PP5/8 b - - 2 50'),
                  Board('8/8/2p2k1p/1p3PK1/p7/P7/1PP5/8 b - - 2 50')]
        # Get a set off all moves from all boards...
        all_moves = set()
        for board in boards:
            all_moves.update(list(board.pseudo_legal_moves))

        best_move = move_strategy.find_best_move_l0(boards, possible_moves=list(all_moves))

        print(f'Best move: {best_move}')

        expected_moves_ucis = [
            'h6h5', 'c6c5', 'f6e5', 'h6g5'
        ]
        unexpected_moves_ucis = [
            'f6f5', 'f6g5'
        ]
        expected_moves = [chess.Move.from_uci(uci) for uci in expected_moves_ucis]
        unexpected_moves = [chess.Move.from_uci(uci) for uci in unexpected_moves_ucis]
        # Assuming that expected_moves and unexpected_moves are defined
        self.assertIn(best_move, expected_moves, msg=f'{best_move} is not in the expected moves.')
        self.assertNotIn(best_move, unexpected_moves, msg=f'{best_move} is in the unexpected moves.')

    def test_case_2(self):
        color = chess.BLACK
        game_information_db = GameInformationDB(color, not color)

        move_strategy = MoveStrategy(game_information_db, logger)

        boards = [Board('4rrk1/p1p3pp/3p4/1p3p2/2bPnN2/2P5/P1Q3PP/2KR1B1R b - - 1 25'),
                  Board('4rrk1/p1p3pp/3p4/1p3p2/1NbPn3/2P5/P1Q3PP/2KR1B1R b - - 1 25'),
                  Board('4rrk1/p1p3pp/3p4/1p3p2/2bPn3/2P5/PNQ3PP/2KR1B1R b - - 1 25'),
                  Board('4rrk1/p1p3pp/3p4/1pN2p2/2bPn3/2P5/P1Q3PP/2KR1B1R b - - 1 25'),
                  Board('4rrk1/p1p3pp/3p4/1p2Np2/2bPn3/2P5/P1Q3PP/2KR1B1R b - - 1 25'),
                  Board('4rrk1/p1p3pp/3p4/1p3p2/2bPn3/2P5/P1Q2NPP/2KR1B1R b - - 1 25')]
        # Get a set off all moves from all boards...
        all_moves = set()
        for board in boards:
            all_moves.update(list(board.pseudo_legal_moves))

        best_move = move_strategy.find_best_move_l0(boards, possible_moves=list(all_moves))

        print(f'Best move: {best_move}')

        expected_moves_ucis = [

        ]
        unexpected_moves_ucis = [
            'c4a2'
        ]
        expected_moves = [chess.Move.from_uci(uci) for uci in expected_moves_ucis]
        unexpected_moves = [chess.Move.from_uci(uci) for uci in unexpected_moves_ucis]
        # Assuming that expected_moves and unexpected_moves are defined
        if expected_moves:
            self.assertIn(best_move, expected_moves, msg=f'{best_move} is not in the expected moves.')
        if unexpected_moves:
            self.assertNotIn(best_move, unexpected_moves, msg=f'{best_move} is in the unexpected moves.')

    def test_case_3(self):
        color = chess.WHITE
        game_information_db = GameInformationDB(color, not color)

        move_strategy = MoveStrategy(game_information_db, logger)

        boards = [HashableBoard('r2q1B1k/pp2pp1p/2bp1bp1/8/2P5/Q7/PP1nBPPP/3R2K1 w - - 0 18'),
                  HashableBoard('r1q2Bk1/pp2pp1p/2bp1bp1/8/2P5/Q7/PP1nBPPP/3R2K1 w - - 0 18'),
                  HashableBoard('r2q1Bk1/p3pp1p/2bp1bp1/1p6/2P5/Q7/PP1nBPPP/3R2K1 w - - 0 18'),
                  HashableBoard('r2q1Bk1/pp2pp1p/2bp2p1/6b1/2P5/Q7/PP1nBPPP/3R2K1 w - - 0 18'),
                  HashableBoard('r3qBk1/pp2pp1p/2bp1bp1/8/2P5/Q7/PP1nBPPP/3R2K1 w - - 0 18'),
                  HashableBoard('r2q1Bk1/pp2pp1p/2bp1b2/6p1/2P5/Q7/PP1nBPPP/3R2K1 w - - 0 18'),
                  HashableBoard('r2q1Bk1/pp2pp1p/2bp2p1/8/2P1n3/Q7/PP1bBPPP/3R2K1 w - - 0 18'),
                  HashableBoard('r2q1Bk1/p3pp1p/1pbp1bp1/8/2P5/Q7/PP1nBPPP/3R2K1 w - - 0 18')]
        # Get a set off all moves from all boards...
        all_moves = set()
        for board in boards:
            all_moves.update(list(board.pseudo_legal_moves))

        best_move = move_strategy.find_best_move_l0(boards, possible_moves=list(all_moves))

        print(f'Best move: {best_move}')

        expected_moves_ucis = [

        ]
        unexpected_moves_ucis = [
            'f8g7'
        ]
        expected_moves = [chess.Move.from_uci(uci) for uci in expected_moves_ucis]
        unexpected_moves = [chess.Move.from_uci(uci) for uci in unexpected_moves_ucis]
        # Assuming that expected_moves and unexpected_moves are defined
        if expected_moves:
            self.assertIn(best_move, expected_moves, msg=f'{best_move} is not in the expected moves.')
        if unexpected_moves:
            self.assertNotIn(best_move, unexpected_moves, msg=f'{best_move} is in the unexpected moves.')

    def test_case_4(self):
        color = chess.WHITE
        game_information_db = GameInformationDB(color, not color)

        move_strategy = MoveStrategy(game_information_db, logger)

        boards = [HashableBoard('1rb1kb1r/1p1p1ppp/4qnn1/p1pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('1rb1kb1r/1p2qppp/p2p1nn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13'),
                  HashableBoard('1rb1kb1r/3p1ppp/1p2qnn1/p1pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('1rb1kb1r/3pqppp/5nn1/pp1P4/2p4P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13'),
                  HashableBoard('1rb1kb1r/4qppp/pp1p1nn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13'),
                  HashableBoard('r1b1kb1r/1p1p1ppp/p3qnn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('r1b1kb1r/1p2qppp/3p1nn1/p1pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13'),
                  HashableBoard('1rb1kb1r/1p1p1ppp/p4nn1/2pP4/7P/P1NQqNP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('1rb1kb1r/3p1ppp/5nn1/pppPq3/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('1rbk1b1r/1p2qppp/3p1nn1/p1pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQ - 0 13'),
                  HashableBoard('1rb1kb1r/1p1p1ppp/4qnn1/2pP4/p6P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('1rb1kb1r/pp1p1ppp/4qnn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQ - 1 13'),
                  HashableBoard('1rbk1b1r/1p1p1ppp/p3qnn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQ - 1 13'),
                  HashableBoard('1rb1kb1r/p2p1ppp/5nn1/1ppPq3/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('1rbk1b1r/pp2qppp/3p1nn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQ - 0 13'),
                  HashableBoard('1rb1kb1r/4qppp/p2p1nn1/1ppP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13'),
                  HashableBoard('1rb1kb1r/pp1pqppp/5nn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 1 13'),
                  HashableBoard('r1b1kb1r/pp2qppp/3p1nn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13'),
                  HashableBoard('1rb1kb1r/1p2qppp/3p1nn1/p1pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13'),
                  HashableBoard('1rb1kb1r/p3qppp/1p1p1nn1/2pP4/7P/P1NQ1NP1/1P2PP2/R1B1KB1R w KQk - 0 13')]
        # Get a set off all moves from all boards...
        all_moves = set()
        for board in boards:
            all_moves.update(list(board.pseudo_legal_moves))

        best_move = move_strategy.find_best_move_l0(boards, possible_moves=list(all_moves))

        print(f'Best move: {best_move}')

        expected_moves_ucis = [

        ]
        unexpected_moves_ucis = [
            'h4h5'
        ]
        expected_moves = [chess.Move.from_uci(uci) for uci in expected_moves_ucis]
        unexpected_moves = [chess.Move.from_uci(uci) for uci in unexpected_moves_ucis]
        # Assuming that expected_moves and unexpected_moves are defined
        if expected_moves:
            self.assertIn(best_move, expected_moves, msg=f'{best_move} is not in the expected moves.')
        if unexpected_moves:
            self.assertNotIn(best_move, unexpected_moves, msg=f'{best_move} is in the unexpected moves.')


if __name__ == '__main__':
    unittest.main()
