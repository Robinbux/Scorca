import chess
import pickle
import time

from src.boards_tracker import BoardsTracker

BOARDS_NAME = '23498_random_chess_boards.pkl'


def load_boards():
    with open(BOARDS_NAME, 'rb') as f:
        boards = pickle.load(f)
    return boards


def test_handle_own_move_result(boards_tracker):
    # Example move result:
    captured_opponent_piece = True
    capture_square = chess.SQUARES[chess.C6]
    requested_move = chess.Move(chess.D5, chess.C6)
    taken_move = chess.Move(chess.D5, chess.C6)

    print(f'Boards before: {len(boards_tracker.possible_states)}')
    start_time = time.perf_counter()
    boards_tracker.handle_own_move_result(captured_opponent_piece, capture_square, requested_move, taken_move, 3)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f'Boards after: {len(boards_tracker.possible_states)}')

def test_handle_opponent_move_result(boards_tracker):
    # Example move result:
    captured_my_piece = False
    capture_square = None

    print(f'Boards before: {len(boards_tracker.possible_states)}')
    start_time = time.perf_counter()
    boards_tracker.handle_opponent_move_result(captured_my_piece, capture_square, 3)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f'Boards after: {len(boards_tracker.possible_states)}')

def test_handle_sense_result(boards_tracker):
    # Example sense result:
    sense_result = [(chess.SQUARES[chess.D5], chess.Piece(chess.PAWN, chess.WHITE)),
                    (chess.SQUARES[chess.D4], chess.Piece(chess.PAWN, chess.WHITE)),
                    (chess.SQUARES[chess.H6], chess.Piece(chess.KNIGHT, chess.BLACK))]

    print(f'Boards before: {len(boards_tracker.possible_states)}')
    start_time = time.perf_counter()
    boards_tracker.handle_sense_result(sense_result, 3)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f'Boards after: {len(boards_tracker.possible_states)}')


if __name__=='__main__':
    boards = load_boards()
    boards = boards + boards
    boards_tracker = BoardsTracker()
    boards_tracker.possible_states = set(boards)
    # test_handle_own_move_result(boards_tracker)
    # test_handle_opponent_move_result(boards_tracker)
    test_handle_sense_result(boards_tracker)
