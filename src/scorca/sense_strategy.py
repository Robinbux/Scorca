from typing import List

import chess
import itertools
import numpy as np
import random
from joblib import Parallel, delayed


np.seterr(all='ignore')

piece_weights = np.array([100, 280, 320, 479, 929, 3000])

piece_weights_dict = {
    'P': 100, 'N': 290, 'B': 310, 'R': 500, 'Q': 900, 'K': 1000,
    'p': 100, 'n': 290, 'b': 310, 'r': 500, 'q': 900, 'k': 1000,
    None: 0
}

char_to_index = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11, None: 12
}

index_to_char = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k', 12: None
}


NEIGHBORHOOD_DELTA = list(itertools.product(range(-1, 2), range(-1, 2)))
INV_LOG2 = 1 / np.log(2)

MAX_BOARDS = 4000

def calculate_entropy_scores(possible_boards):
    n_boards = len(possible_boards)
    piece_counts = np.zeros((8, 8, 13))
    reciprocal_boards = 1.0 / n_boards

    for board in possible_boards:
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            piece_index = char_to_index[piece.symbol()] if piece else char_to_index[None]
            piece_counts[square // 8, square % 8, piece_index] += 1

    piece_probs = piece_counts * reciprocal_boards
    entropy = -np.nansum(piece_probs * np.log2(piece_probs), axis=2) * INV_LOG2
    entropy[np.isnan(entropy)] = 0

    return entropy, piece_probs


def calculate_threat_weights(board):
    threat_weights = np.zeros(64)
    color = board.turn
    for piece_type in range(chess.KING, chess.PAWN - 1, -1):
        for piece in board.pieces(piece_type, color):
            for attacker in board.attackers(not color, piece):
                threat_weights[attacker] += piece_weights[piece_type - 1]
    return threat_weights


class NaiveEntropySense:

    def __init__(self, move_strategy, include_penalty=False, include_ponder_bonus=False, include_attacker_squares=False, include_piece_value_bonus=True, ):
        self.move_strategy = move_strategy
        self.sensed_squares = np.zeros((8, 8), dtype=int)
        self.include_penalty = include_penalty
        self.include_ponder_bonus = include_ponder_bonus
        self.include_attacker_squares = include_attacker_squares
        self.include_piece_value_bonus = include_piece_value_bonus
        self.count = 0

    def sense(self, possible_boards: List[chess.Board], ponder_moves: List[chess.Move], game_information_db) -> int:
        if game_information_db.turn == 0 and game_information_db.color == chess.BLACK:
            return 20 # e3
        if len(possible_boards) > 10000:
            possible_boards = random.sample(possible_boards, 10000)
        # if len(possible_boards) < 100:
        #     square = self.most_likely_move_sense(possible_boards)
        square = self.find_highest_entropy_region_square(possible_boards, ponder_moves)
        # Update the sensed squares
        # if self.include_penalty:
        #     self.update_sensed_squares(square)
        return square

    def find_highest_entropy_region_square(self, possible_boards, ponder_moves):
        entropy_scores = self.entropy_score(possible_boards, ponder_moves)

        # Apply penalty to entropy scores based on previously sensed squares
        if self.include_penalty:
            decay_factor = 0.5
            entropy_scores -= decay_factor * self.sensed_squares

        # Calculate the sum of entropy scores for all 3x3 regions
        max_sum = 0
        max_square = 12

        for rank in range(1, 7):
            for file in range(1, 7):
                square_sum = entropy_scores[rank - 1:rank + 2, file - 1:file + 2].sum()
                if square_sum > max_sum:
                    max_sum = square_sum
                    max_square = chess.square(file, rank)
        # Return the center field of the 3x3 region with the highest sum
        return max_square

    def entropy_score(self, possible_boards, ponder_moves):
        n_boards = len(possible_boards)
        piece_counts = np.zeros((64, 13))
        attack_counts = np.zeros(64)

        for board in possible_boards:
            for square in range(64):
                piece = board.piece_at(square)
                if piece is not None:
                    piece_type = piece.piece_type
                    color = int(piece.color)
                    piece_index = (piece_type - 1) + color * 6
                else:
                    piece_index = 12
                piece_counts[square, piece_index] += 1

            # Increase attack counts for potential king attack squares
            if self.include_attacker_squares:
                king_square = board.king(board.turn)
                if king_square is not None:
                    attackers = board.attackers(not board.turn, king_square)
                    for square in chess.SQUARES:
                        if square in attackers:
                            # print("ATTACKER SQUARE *********************************")
                            # print(square)
                            attack_counts[square] += 1

        piece_probs = piece_counts / n_boards
        entropy = -np.nansum(piece_probs * np.log2(piece_probs), axis=1)
        entropy[np.isnan(entropy)] = 0

        # Increase entropy of potential king attack squares
        if self.include_attacker_squares:
            attack_probs = attack_counts / n_boards
            attack_bonus_factor = 600_000
            # only apply bonus if entropy is not 0
            print(attack_probs)
            entropy *= (1 + attack_probs)
            #entropy += np.where(entropy > 0, attack_bonus_factor * attack_probs, 0)

        # Add piece value bonus
        if self.include_piece_value_bonus:
            for square in range(64):
                for piece_type, count in enumerate(piece_counts[square, :12]):
                    piece_type_char = index_to_char[piece_type]
                    if count > 0:
                        entropy[square] *= piece_weights_dict[piece_type_char] * count

        self.count += 1
        #print(entropy.reshape((8, 8)))
        return entropy.reshape((8, 8))
