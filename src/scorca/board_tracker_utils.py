from typing import List, Tuple, Optional

import chess
import chess.svg
from concurrent.futures import ProcessPoolExecutor
from reconchess.utilities import capture_square_of_move

from utils import pseudo_legal_moves_with_castling_through_check


def legal_board_for_sense_result(args: Tuple[chess.Board, List[Tuple[chess.Square, Optional[chess.Piece]]]]) -> Optional[chess.Board]:
    board, sense_result = args

    # Turning sense_result into a dictionary for faster lookups
    sense_result_dict = dict(sense_result)

    return next(
        (
            None
            for square, value in sense_result_dict.items()
            if board.piece_at(square) != value
        ),
        board,
    )

def legal_board_for_own_move_result(args: Tuple[chess.Board, Optional[chess.Move], Optional[chess.Move], bool, Optional[chess.Square]]) -> Optional[chess.Board]:
    board, requested_move, taken_move, captured_opponent_piece, capture_square = args

    if requested_move is None:
        board.turn = not board.turn
        return board

    pseudo_legal_moves = pseudo_legal_moves_with_castling_through_check(board)
    if taken_move not in pseudo_legal_moves:
        return None

    is_capture = board.is_capture(taken_move)
    is_en_passant = board.is_en_passant(taken_move)

    if captured_opponent_piece:
        if not is_capture or (not is_en_passant and board.piece_at(capture_square).piece_type == chess.KING):
            return None
    elif is_capture:
        return None

    if requested_move in pseudo_legal_moves and requested_move != taken_move:
        return None

    board.push(taken_move)
    return board


def next_possible_board_states_based_on_opponent_move_result(args: Tuple[chess.Board, bool, Optional[chess.Square]]) -> List[chess.Board]:
    board, captured_my_piece, capture_square = args
    next_boards = []
    for move in pseudo_legal_moves_with_castling_through_check(board):
        if captured_my_piece:
            if not board.is_capture(move) or capture_square_of_move(board, move) != capture_square:
                continue
        elif board.is_capture(move):
            continue

        board_copy = board.copy()
        board_copy.push(move)
        next_boards.append(board_copy)
    return next_boards


def pseudo_legal_moves_with_castling_through_check_list(boards: List[chess.Board]) -> List[chess.Move]:
    with ProcessPoolExecutor() as executor:
        all_moves = list(executor.map(pseudo_legal_moves_with_castling_through_check, boards))
        # Flatten the list of lists into a single list
        all_moves = [move for moves in all_moves for move in moves]
        return list(set(all_moves))
