from io import TextIOWrapper
import chess.pgn
import sys


def parse_tc(tc_string):
    tc_string = tc_string.split("+")
    if len(tc_string) == 1:
        return None
    return str(int(tc_string[0])+40*int(tc_string[1])) + '\n'

def parse_res(res_string):
    res_string = res_string.split("-")
    if ("1/2" in res_string):
        return "0\n"
    return str(int(res_string[0])-int(res_string[1])) + '\n'

def parse_game(game : chess.pgn.Game, out : TextIOWrapper):
    if (game != None):
        tc = parse_tc(game.headers["TimeControl"])
        if (tc == None): return
        out.write(parse_res(game.headers["Result"]))
        out.write(game.headers["WhiteElo"] + '\n')
        out.write(game.headers["BlackElo"] + '\n')
        out.write(tc)

    while (game != None):
        out.write(game.board().fen() + '\n')
        if game.eval():
            out.write(str((2*game.eval().wdl().white().expectation())-1.0))
        else:
            out.write("0")
        out.write('\n')
        game = game.next()
    out.write('\n')

if (len(sys.argv) != 3):
    print("Wrong arguments!\nargs: path-to-input-pgn, path-to-output\n")
    exit(1)

pgn = open(sys.argv[1])
output = open(sys.argv[2], 'a+')
i = 0
game = chess.pgn.read_game(pgn)
while game != None:
    game = chess.pgn.read_game(pgn)
    parse_game(game, output)
    i += 1
print(i)
