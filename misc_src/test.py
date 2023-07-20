import chess

NULL_MOVE = chess.Move.null()

board = chess.Board("rnb1kbnr/pppppppp/8/8/8/6q1/PPPPP2P/RNBQK2R w KQkq - 0 1")
castling_rights = board.castling_rights
board.occupied_co[chess.WHITE] & (chess.BB_F1 | chess.BB_G1)
if castling_rights & chess.BB_H1:
    print("YUP")

