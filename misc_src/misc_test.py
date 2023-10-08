import chess

moves = {(59, False): 864, (51, False): 1216, (43, False): 114, (42, False): 354, (32, False): 1039, (37, False): 416, (36, False): 206, (25, False): 379, (39, False): 116, (44, False): 153, (53, False): 257, (41, False): 349, (45, False): 554, (46, False): 480}

for move, score in moves.items():
    move_square = chess.square_name(move[0])
    print(move_square)