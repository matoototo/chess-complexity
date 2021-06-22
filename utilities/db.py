import sqlite3

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

        # Table for saving positions and their Elo as determined by binary search
        self.cur.execute(
        f"""CREATE TABLE IF NOT EXISTS calibrated_positions (
            FEN TEXT,
            Elo REAL,
            eval REAL,
            threshold REAL
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
