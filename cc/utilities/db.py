import sqlite3
import json

class PuzzleDatabase:
    """A class for storing and retrieving positions.
        ## Attribute descriptions
        - FEN - FEN string
        - Elo - Elo given as input to the model
        - eval - Computer evaluation of the position, scaled to WR
        - err - Human error, calculated as eval(t+1)-eval(t)
        - pred_err - The predicted error by the model
        - player - Player username
        - threshold - The pred_err according to which the Elo of the position was determined"""

    def __init__(self, path):
        """Creates (or opens if exists) a database at given path."""
        self.con = sqlite3.connect(path)
        self.cur = self.con.cursor()

        # Table for saving positions and their Elo as determined by binary search
        self.cur.execute(
        f"""CREATE TABLE IF NOT EXISTS puzzles (
            id INTEGER PRIMARY KEY,
            FEN TEXT,
            Elo REAL,
            eval REAL,
            threshold REAL,
            UNIQUE(FEN, threshold)
        )""")

        # Table for saving the top 10 moves (according to multipv at depth 20) and their evaluations
        self.cur.execute(
        f"""CREATE TABLE IF NOT EXISTS moves (
            id INTEGER PRIMARY KEY,
            puzzle_id INTEGER REFERENCES puzzles(id) ON DELETE CASCADE,
            move STRING,
            eval REAL
        )""")

        # Table for saving players and their puzzle Elo
        self.cur.execute(
        f"""CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            username TEXT,
            Elo REAL DEFAULT 1500.0,
            RD REAL DEFAULT 200.0,
            vol REAL DEFAULT 0.06,
            UNIQUE(username)
        )""")

        # Table for saving the player puzzle attempts
        self.cur.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
            player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
            puzzle_id INTEGER REFERENCES puzzles(id) ON DELETE CASCADE,
            delta REAL,
            PRIMARY KEY(player_id, puzzle_id)
        )""")

        # Table for saving data about users
        self.cur.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL UNIQUE,
            email VARCHAR NOT NULL UNIQUE,
            password VARCHAR NOT NULL,
            lichess VARCHAR UNIQUE
        )""")

    def insert_json(self, path, threshold):
        """Inserts puzzles and moves found in the given .json to the database, with a given threshold.
            Format of the .json is given by assign_elo.py."""
        get_eval = lambda fen: moves[fen][0]['eval']
        puzzles = json.load(open(path, 'r'))
        moves = { x['fen'] : x['moves'] for x in puzzles }
        puzzles = [(x['fen'], x['elo'], get_eval(x['fen']), threshold) for x in puzzles]
        for puzzle in puzzles:
            self.cur.execute("INSERT OR IGNORE INTO puzzles VALUES (NULL, ?, ?, ?, ?)", puzzle)
            puzzle_id = self.cur.lastrowid
            if self.cur.rowcount > 0:
                insert_moves = [(puzzle_id, move['move'], move['eval']) for move in moves[puzzle[0]]]
                self.cur.executemany("INSERT OR IGNORE INTO moves VALUES (NULL, ?, ?, ?)", insert_moves)
        self.con.commit()


class PositionDatabase:
    """A class for storing and retrieving positions.
        ## Attribute descriptions
        - FEN - FEN string
        - Elo - Elo given as input to the model
        - eval - Computer evaluation of the position, scaled to WR
        - err - Human error, calculated as eval(t+1)-eval(t)
        - pred_err - The predicted error by the model
        - player - Player username
        - threshold - The pred_err according to which the Elo of the position was determined"""

    def __init__(self, path):
        """Creates (or opens if exists) a database at given path."""
        self.con = sqlite3.connect(path)
        self.cur = self.con.cursor()

        # Dummy ? table that can save the positions given by lichess.py
        self.cur.execute(
        f"""CREATE TABLE IF NOT EXISTS player_positions (
            FEN TEXT,
            Elo REAL,
            eval REAL,
            err REAL,
            pred_err REAL,
            player TEXT,
            game_id TEXT,
            PRIMARY KEY(FEN, game_id)
        )""")

    def insert_player_positions(self, positions):
        """Positions should be a List of 3-tuples with the values: error, predicted error, Board object."""
        processed = [(x[2].fen, self.__player_elo(x[2]), x[2].eval, x[0], x[1], self.__player_name(x[2]), x[2].game.id) for x in positions]
        self.cur.executemany("INSERT OR IGNORE INTO player_positions VALUES (?, ?, ?, ?, ?, ?, ?)", processed)
        self.con.commit()

    def __player_elo(self, board):
        return board.game.welo if board.side_to_move() == 1.0 else board.game.belo

    def __player_name(self, board):
        return board.game.white if board.side_to_move() == 1.0 else board.game.black
