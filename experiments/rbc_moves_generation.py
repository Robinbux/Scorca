import chess
import unittest
from opponent_bots.strangefish.utilities import rbc_legal_moves


def old_rbc_moves(board):
    moves = rbc_legal_moves(board)
    return moves

def generate_rbc_legal_moves(board):
    """
    Generates all legal moves for a given chess.Board object, but for Reconnaissance Blind Chess (RBC).

    :param board: A chess.Board object representing the current state of the chessboard
    :return: A list of legal moves in RBC
    """
    legal_moves = []

    # Occupancy bitboard with the squares occupied by either color
    occupancy = board.occupied

    # Generate moves for each piece type
    for piece_type in chess.PIECE_TYPES:
        # Get the bitboard of pieces of the current type for the current player
        pieces = board.pieces(piece_type, board.turn)

        # Generate moves for each piece
        for from_square in chess.scan_forward(pieces):
            # Use internal attack generator
            attacks = board.attacks_mask(from_square)

            # If the piece is a pawn, we need to make sure it doesn't capture vertically
            if piece_type == chess.PAWN:
                attacks &= ~board.pawns

            # The legal targets are the intersection of the attacks and the non-occupied squares
            targets = attacks & ~occupancy

            # Add moves to the legal moves list
            for to_square in chess.scan_forward(targets):
                move = chess.Move(from_square, to_square)

                # We need to handle promotions for pawns separately
                if piece_type == chess.PAWN and chess.BB_SQUARES[to_square] & (chess.BB_RANK_1 | chess.BB_RANK_8):
                    legal_moves.extend([move._replace(promotion=promotion) for promotion in chess.PROMOTIONS])
                else:
                    legal_moves.append(move)

    return legal_moves

class TestGenerateRBCMoves(unittest.TestCase):

    def test_generate_rbc_legal_moves(self):
        test_boards = [
            # Initial position
            {
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'expected_num_moves': 20
            },
            # Position with pieces close to each other
            {
                'fen': '8/8/3k4/3Pp3/3pP3/3K4/8/8 w - - 0 1',
                'expected_num_moves': 7
            },
            # Position where a pawn can be promoted
            {
                'fen': '7k/3P4/8/8/8/8/8/7K w - - 0 1',
                'expected_num_moves': 4
            },
            # Completely empty board (no legal moves)
            {
                'fen': '8/8/8/8/8/8/8/8 w - - 0 1',
                'expected_num_moves': 0
            },
            # Only kings on the board
            {
                'fen': '8/8/8/4k3/8/4K3/8/8 w - - 0 1',
                'expected_num_moves': 8
            },
            # Random position
            {
                'fen': 'rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq d6 0 2',
                'expected_num_moves': 30
            }
        ]

        for test_board in test_boards:
            with self.subTest(fen=test_board['fen']):
                board = chess.Board(test_board['fen'])
                legal_moves = generate_rbc_legal_moves(board)
                self.assertEqual(len(legal_moves), test_board['expected_num_moves'])


if __name__ == '__main__':
    unittest.main()