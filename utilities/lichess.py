import sys, os, io

sys.path.append(sys.path.append(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parser')))

from utilities.pgn_to_data import parse_pgn
from utilities.API_TOKEN import API_TOKEN
from data import InferDataset
from db import PositionDatabase

import chess
import berserk
import berserk.utils
import utilities.evaluator as evaluator
from datetime import date, datetime

def get_game(id, token):
    """Returns the PGN of a game of the given Lichess game ID."""
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    return client.games.export(id, as_pgn=True, evals="true")

def player_games_since(username, token, since : datetime, only_analysed = None):
    """Returns a generator of PGNs of games played by some player since some datetime.
    To filter games with no analysis, use the only_analysed attribute (defaults to None)."""
    only_analysed = None if only_analysed in [None, False] else "true" # berserk is broken, must cast to string manually...
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    return client.games.export_by_player(username, as_pgn=True, since=int(berserk.utils.to_millis(since)), analysed=only_analysed, evals="true")

def get_game_data(id, token, engine: chess.engine.SimpleEngine = None, depth = 20, zero_first = True, default_elo = 1500):
    """Returns List of Boards of the given Lichess game ID."""
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    return parse_pgn(io.StringIO(client.games.export(id, as_pgn=True, evals="true")), engine, depth, zero_first, default_elo)

def eval_game_and_sort(pgn, net, engine: chess.engine.SimpleEngine = None, depth = 20, zero_first = True, default_elo = 1500, sort = True):
    """Evaluates a given PGN with a model and sorts the positions by largest delta between expected and actual error.
    Returns a List of 3-tuples: error, predicted_error, Board."""
    data = parse_pgn(io.StringIO(pgn), engine, depth, zero_first, default_elo)
    evaluated = evaluator.eval_data(net, InferDataset(data))
    labeled = []
    for i in range(len(evaluated)-1):
        error = (evaluated[i][1].eval - evaluated[i+1][1].eval)*evaluated[i][1].side_to_move()
        labeled.append((error, *evaluated[i]))
    if sort: labeled.sort(key = lambda x : x[0]-x[1])
    return labeled

def name_filter(board, name):
    """Filter predicate used to filter out those positions where name is not to-move."""
    return (board.game.white == name and board.side_to_move() == 1.0) or \
           (board.game.black == name and board.side_to_move() == -1.0)

