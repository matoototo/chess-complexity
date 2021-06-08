import sys, os, io

sys.path.append(sys.path.append(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parser')))

from utilities.pgn_to_data import parse_pgn
from utilities.API_TOKEN import API_TOKEN
from data import InferDataset

import chess
import berserk
import berserk.utils
import utilities.evaluator as evaluator
from datetime import date, datetime

def get_game(id, token):
    """Returns the PGN of a game of the given Lichess game ID."""
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    return client.games.export(id, as_pgn=True)

# only_analysed doesn't work for whatever reason. berserk bug?
def player_games_since(username, token, since : datetime, only_analysed = False):
    """Returns a generator of PGNs of games played by some player since some datetime.
    To filter games with no analysis, use the only_analysed attribute (defaults to False)."""
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    return client.games.export_by_player(username, as_pgn=True, since=berserk.utils.to_millis(since), analysed=only_analysed)

def get_game_data(id, token, engine: chess.engine.SimpleEngine = None, depth = 20, zero_first = True, default_elo = 1500):
    """Returns List of Boards of the given Lichess game ID."""
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    return parse_pgn(io.StringIO(client.games.export(id, as_pgn=True)), engine, depth, zero_first, default_elo)

def eval_game_and_sort(pgn, net, engine: chess.engine.SimpleEngine = None, depth = 20, zero_first = True, default_elo = 1500):
    """Evaluates a given PGN with a model and sorts the positions by largest delta between expected and actual error.
    Returns a List of 5-tuples: delta, eval, error, predicted_error, FEN."""
    data = parse_pgn(io.StringIO(pgn), engine, depth, zero_first, default_elo)
    evaluated = evaluator.eval_data(net, InferDataset(data))
    labeled = []
    for i in range(len(evaluated)-1):
        error = (evaluated[i][1].eval - evaluated[i+1][1].eval)*evaluated[i][1].side_to_move()
        delta = error - evaluated[i][0]
        labeled.append((delta, evaluated[i][1].eval, error, *evaluated[i]))
    labeled.sort(key = lambda x : x[0])
    return labeled


