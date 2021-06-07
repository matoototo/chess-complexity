import sys, os, io

sys.path.append(sys.path.append(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parser')))

from utilities.pgn_to_data import parse_pgn
from utilities.API_TOKEN import API_TOKEN
from data import InferDataset

import berserk
import utilities.evaluator as evaluator

def get_game_data(id, token):
    """Returns List of Boards of the given Lichess game ID."""
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    return parse_pgn(io.StringIO(client.games.export(id, as_pgn=True)))

