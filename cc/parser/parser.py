import os
import time
import argparse
import chess.pgn
import functools
from pathlib import Path
from io import TextIOWrapper
import multiprocessing as mp


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

    while (game != None):
        out.write(game.board().fen() + '\n')
        if game.eval():
            out.write(str((2*game.eval().wdl(model="sf15.1").white().expectation())-1.0))
        else:
            out.write("0")
        out.write('\n')
        game = game.next()
    out.write('\n')

def parse(pgn_file, input_dir, output_dir):
    print(pgn_file, flush=True)
    split = pgn_file.split(".")[0].split("_")

    prefix = "" if len(split) == 2 else split[-3]
    filename = f"{prefix}_processed_{split[-1]}.data"
    if filename in os.listdir(output_dir): return

    i = 0
    filepath = output_dir / filename
    with open(input_dir / pgn_file) as pgn, open(filepath, 'a+') as output:
        game = chess.pgn.read_game(pgn)
        while game != None:
            parse_game(game, output)
            game = chess.pgn.read_game(pgn)
            i += 1
    print(i, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input pgn", type=Path)
    parser.add_argument("--output", required=True, help="path to output", type=Path)
    parser.add_argument("--threads", required=False, help="number of threads to spawn (default = core count)", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    start = time.time()
    pool = mp.Pool(args.threads)
    pgn_files = [f for f in os.listdir(args.input) if f.endswith(".pgn")]

    job = functools.partial(parse, input_dir=args.input, output_dir=args.output)
    pool.map(job, pgn_files)
    print("Time: ", time.time() - start)
