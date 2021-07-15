import sys, os

import cc.parser.data as data

import chess.pgn
import chess.engine
import chess

def transform_eval(score):
    return (2*score.wdl(model="lichess").white().expectation())-1.0

def eval_fen(fen, engine: chess.engine.SimpleEngine, depth):
    if engine == None:
        raise AttributeError("The passed position(s) don't have an evaluation, but an engine for evaluation was not defined.")
    eval = transform_eval(engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))["score"])
    return eval

def parse_tc(tc_string):
    tc_string = tc_string.split("+")
    if len(tc_string) == 1:
        return None
    return int(tc_string[0])+40*int(tc_string[1])

def parse_res(res_string):
    res_string = res_string.split("-")
    if ("*" in res_string):
        return None
    if ("1/2" in res_string):
        return 0
    return int(res_string[0])-int(res_string[1])

def parse_pgn(pgn, engine: chess.engine.SimpleEngine = None, depth = 20, zero_first = True, default_elo = 1500, default_tc = 600):
    """Constructs a List of Boards from the given PGN.
    If evaluations are not given in the PGN expects engine to be defined,
    which will then be used to evaluate the position(s) at specified depth.
    Due to a bug in python-chess, the startpos does not have an evaluation.
    By default it is set to 0.0, which can be avoided with the zero_first flag.
    If zero_first is set to False the method requires a valid engine attribute, as it will have to evaluate the position.
    Unknown Elo ('?') are replaced by default_elo, which is by default 1500.
    Likewise, unknown time controls are replaced by default_tc, which is by default 600.
    """
    game = chess.pgn.read_game(pgn)
    positions = []
    tc = parse_tc(game.headers["TimeControl"])
    tc = tc if tc != None else default_tc
    res = parse_res(game.headers["Result"])
    welo = game.headers["WhiteElo"]
    belo = game.headers["BlackElo"]
    welo = int(welo) if welo != "?" else default_elo
    belo = int(belo) if belo != "?" else default_elo
    white = game.headers["White"]
    black = game.headers["Black"]

    g = data.Game(res, welo, belo, tc, white, black)

    first = True
    while (game != None):
        fen = game.board().fen()
        if game.eval():
            eval = (2*game.eval().wdl(model="lichess").white().expectation())-1.0
        else:
            if first and zero_first:
                first = not first
                eval = 0.0
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
