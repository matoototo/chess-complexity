import os
import time
import argparse
import chess.pgn
import functools
from pathlib import Path
from io import TextIOWrapper
import multiprocessing as mp

def base_time(tc):
    return int(tc[0])

def increment(tc):
    return int(tc[1])

def parse_tc(tc_string):
    tc = tc_string.split("+")
    if len(tc) == 1:
        return None
    return str(base_time(tc)+40*increment(tc)) + '\n'

def parse_res(res_string):
    res_string = res_string.split("-")
    if ("*" in res_string):
        return None
    if ("1/2" in res_string):
        return "0\n"
    return str(int(res_string[0])-int(res_string[1])) + '\n'

def time_used(game : chess.pgn.Game, white_time, black_time, inc):
    is_white = game.turn()

    previous_time = white_time if is_white else black_time
    current_time = game.clock()

    if not current_time:
        return white_time, black_time, 0
        raise RuntimeError(f"Clock data not available. {game} {white_time} {black_time} {inc}")

    white_time = white_time if not is_white else current_time
    black_time = black_time if is_white else current_time

    time_used = int(previous_time + inc - current_time)
    return white_time, black_time, time_used

def parse_game(game : chess.pgn.Game, out : TextIOWrapper):
    if (game != None):
        tc = parse_tc(game.headers["TimeControl"])
        res = parse_res(game.headers["Result"])
        if (tc == None or res == None): return
        out.write(res)
        out.write(game.headers["WhiteElo"] + '\n')
        out.write(game.headers["BlackElo"] + '\n')
        out.write(tc)


        tc_list = game.headers["TimeControl"].split("+")
        white_time = base_time(tc_list)
        black_time = base_time(tc_list)
        inc = increment(tc_list)

    while (game != None):
        if game.clock():
            white_time, black_time, used_time = time_used(game, white_time, black_time, inc)
            out.write(f"{used_time}\n")

        out.write(game.board().fen() + '\n')
        if game.eval():
            out.write(str((2*game.eval().wdl(model="sf15.1").white().expectation())-1.0))
        else:
            out.write("0")
        out.write('\n')

        game = game.next()
    out.write("0\n")
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
