import chess.pgn
import sys

if (len(sys.argv) != 2):
    print("Wrong arguments!\nargs: path-to-pgn\n")
    exit(1)

pgn = open(sys.argv[1])
i = 0
game = chess.pgn.read_game(pgn)
while game != None:
    game = chess.pgn.read_game(pgn)
    i += 1
print(i)
