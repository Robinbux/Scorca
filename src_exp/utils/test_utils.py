from unittest import TestCase

from src.utils import get_best_center_from_best_op_moves_dict, extend_sliding_move

from lczero.backends import Backend, Weights, GameState
import chess


class Test(TestCase):

    def test_get_best_center_from_best_op_moves_dict(self):
        squares = {(59, 51, False): 63, (23, 14, True): 19, (59, 58, False): 24, (25, 10, True): 16, (61, 52, False): 4,
                   (61, 34, False): 63, (33, 25, False): 3, (50, 34, False): 12, (37, 10, True): 17, (53, 37, False): 3,
                   (55, 39, False): 11, (60, 62, False): 12, (50, 42, False): 29, (49, 33, False): 1, (59, 35, True): 9,
                   (62, 45, False): 3, (57, 42, False): 2, (27, 10, True): 17, (39, 45, False): 2, (46, 10, True): 1,
                   (28, 18, True): 3, (28, 38, False): 2, (30, 12, True): 4, (25, 18, True): 5, (39, 22, True): 3,
                   (61, 43, False): 6, (30, 20, False): 3, (45, 35, True): 1, (30, 13, True): 1, (33, 18, True): 1,
                   (48, 40, False): 1, (16, 52, False): 1, (16, 34, False): 4, (57, 40, False): 5, (16, 25, False): 1,
                   (54, 46, False): 1, (28, 22, True): 1, (28, 43, False): 1, (61, 25, False): 7, (27, 12, True): 3,
                   (37, 46, False): 1, (28, 10, True): 1, (28, 14, True): 1, (28, 46, False): 1, (54, 38, False): 2,
                   (23, 37, False): 1, (40, 25, False): 1, (34, 13, True): 2, (62, 63, False): 1, (16, 9, True): 1,
                   (39, 31, False): 4, (25, 34, False): 2, (39, 46, False): 1, (48, 33, False): 1, (30, 37, False): 1,
                   (61, 54, False): 2, (55, 47, False): 1, (47, 39, False): 1, (39, 29, False): 1, (30, 45, False): 2,
                   (36, 28, False): 3, (21, 14, True): 1, (21, 12, True): 1, (19, 10, True): 1}
        center_square = get_best_center_from_best_op_moves_dict(squares)
        print(center_square)
        self.fail()

    def test_l0_eval(self):
        pass


def l0_eval(board: chess.Board):
    # Load weights
    w = Weights()

    # Choose a backend
    b = Backend(weights=w)

    print(Backend.available_backends())


class Test(TestCase):
    def test_extend_sliding_move(self):
        move = chess.Move.from_uci("e1h1")
        extended_move = extend_sliding_move(move)
        print(extended_move)
