import sys, os

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parser')))

import data

import chess.pgn
import chess.engine
import chess

def transform_eval(score):
    return str((2*score.wdl(model="lichess").white().expectation())-1.0)

def eval_fen(fen, engine: chess.engine.SimpleEngine, depth):
    eval = transform_eval(engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))["score"])
    return eval

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

def parse_pgn(pgn, engine: chess.engine.SimpleEngine = None, depth = 20, zero_first = True):
    """Constructs a List of Boards from the given PGN.
    If evaluations are not given in the PGN expects engine to be defined,
    which will then be used to evaluate the position(s) at specified depth.
    Due to a bug in python-chess, the startpos does not have an evaluation.
    By default it is set to 0.0, which can be avoided with the zero_first flag.
    If zero_first is set to False the method requires a valid engine attribute, as it will have to evaluate the position.
    """
    game = chess.pgn.read_game(pgn)
    positions = []
    tc = parse_tc(game.headers["TimeControl"])
    res = parse_res(game.headers["Result"])
    welo = game.headers["WhiteElo"]
    belo = game.headers["BlackElo"]

    g = data.Game(res, welo, belo, tc)

    first = True
    while (game != None):
        fen = game.board().fen()
        if game.eval():
            eval = str((2*game.eval().wdl(model="lichess").white().expectation())-1.0)
        else:
            if first and zero_first:
                first = not first
                eval = "0.0"
            else:
                eval = eval_fen(fen, engine, depth)
        positions.append(data.Board(g, fen, eval))
        game = game.next()
    return positions

def parse_fen(fen, welo, belo, tc, res, eval = None, engine: chess.engine.SimpleEngine = None, depth = 20):
    """Constructs a Board object given fen and attributes.
    If an eval is not given expects engine to be defined,
    which will then be used to evaluate the position at specified depth.
    The evaluation is expected to be first transformed with the transform_eval method.
    """
    if (not eval): eval = eval_fen(fen, engine, depth)
    return data.Board(data.Game(res, welo, belo, tc), fen, eval)
