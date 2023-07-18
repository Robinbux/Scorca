from typing import List

import chess
import chess.pgn
import io

# Path to your PGN test suite file
pgn_file_path = '/Users/robinbux/Desktop/RBC_New/misc_src/chess_positions_test_suite.pgn'

def load_chess_boards() -> List[chess.Board]:
    # Read the PGN file
    with open(pgn_file_path, 'r') as pgn_file:
        pgn_content = pgn_file.read()

    # Use StringIO to treat the string as a file
    pgn_io = io.StringIO(pgn_content)

    # List to store chess.Board objects
    chess_boards = []

    # Iterate through all games in the PGN file
    while True:
        # Read the next game from the PGN file
        game = chess.pgn.read_game(pgn_io)

        # Break the loop if there are no more games
        if game is None:
            break

        # Get the board from the game and add it to the list
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
        chess_boards.append(board)

    return chess_boards
