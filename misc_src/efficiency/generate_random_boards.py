import chess
import random
import pickle
from tqdm import tqdm

AMOUNT_OF_BOARDS = 10000
AMOUNT_OF_RANDOM_MOVES = 30

def generate_random_move(board):
    return random.choice(list(board.legal_moves))

boards = []
for _ in tqdm(range(AMOUNT_OF_BOARDS)):
    board = chess.Board()
    for _ in range(AMOUNT_OF_RANDOM_MOVES):
        if not board.is_game_over():
            move = generate_random_move(board)
            board.push(move)
    if not board.is_game_over():
        boards.append(board)

with open(f'{AMOUNT_OF_BOARDS}_random_chess_boards.pkl', 'wb') as f:
    pickle.dump(boards, f)