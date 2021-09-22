from io import TextIOWrapper
import chess.pgn
import sys
import os


def parse_tc(tc_string):
    tc_string = tc_string.split("+")
    if len(tc_string) == 1:
        return None
    return str(int(tc_string[0])+40*int(tc_string[1])) + '\n'

def parse_res(res_string):
    res_string = res_string.split("-")
    if ("*" in res_string):
        return None
    if ("1/2" in res_string):
        return "0\n"
    return str(int(res_string[0])-int(res_string[1])) + '\n'

def parse_game(game : chess.pgn.Game, out : TextIOWrapper):
    if (game != None):
        tc = parse_tc(game.headers["TimeControl"])
        res = parse_res(game.headers["Result"])
        if (tc == None or res == None): return
        out.write(res)
        out.write(game.headers["WhiteElo"] + '\n')
        out.write(game.headers["BlackElo"] + '\n')
        out.write(tc)

    last_w = int(game.headers["TimeControl"].split("+")[0])
    last_b = int(game.headers["TimeControl"].split("+")[0])
    increment = int(game.headers["TimeControl"].split("+")[1])
    while (game != None):
        out.write(game.board().fen() + '\n')
        if game.eval():
            out.write(str((2*game.eval().wdl(model="lichess").white().expectation())-1.0))
        else:
            out.write("0")
        out.write('\n')
        if game.clock():
            out.write(str((last_w if game.turn() else last_b) + increment-game.clock()))
            if game.turn(): last_w = game.clock()
            else: last_b = game.clock()
        else:
            out.write("0")
        out.write('\n')
        game = game.next()
    out.write('\n')

if (len(sys.argv) != 3):
    print("Wrong arguments!\nargs: path-to-input-pgn, path-to-output\n")
    exit(1)

for pgn_file in os.listdir(sys.argv[1]):
    print(pgn_file)
    split = pgn_file.split(".")[0].split("_")
    filename = sys.argv[2] + split[-3] + "_processed_" + split[-1] + ".data"
    if filename.split('/')[-1] in os.listdir(sys.argv[2]): continue
    pgn = open(sys.argv[1] + pgn_file)
    output = open(filename, 'a+')
    i = 0
    game = chess.pgn.read_game(pgn)
    while game != None:
        parse_game(game, output)
        game = chess.pgn.read_game(pgn)
        i += 1
    print(i)
