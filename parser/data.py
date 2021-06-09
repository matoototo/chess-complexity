import torch
import torch.utils.data
import numpy as np
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader


def create_offsets():
    offset = defaultdict(int)
    for char, i in zip(['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P'], range(12)):
        offset[char] = i*64
    return offset

offsets = create_offsets()


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, positions = []):
        self.positions = np.array(positions)

    def __getitem__(self, i):
        """Returns the planes and their index in self.positions."""
        return (self.positions[i].to_planes(), i)

    def __len__(self):
        return len(self.positions)


class PositionDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename)
        self.positions = []
        self.labels = []

    def __getitem__(self, i):
        return (self.positions[i].to_planes(), self.labels[i])

    def __len__(self):
        return len(self.positions)

    def parse_data(self, limit = None):
        self.positions = []
        self.labels = []
        while True:
            line = self.file.readline().split(',')
            if (len(line) < 7): break
            fen = line[0]
            eval = float(line[1])
            eval_next = float(line[2])
            winner = int(line[3])
            welo = int(line[4])
            belo = int(line[5])
            tc = int(line[6])
            if (limit and len(self.positions) >= limit): break
            game = Game(winner, welo, belo, tc)
            self.positions.append(Board(game, fen, eval))
            self.labels.append(eval_delta(eval, eval_next, self.positions[-1].side_to_move()))
        self.positions = np.array(self.positions)
        self.labels = np.array(self.labels).reshape(len(self.labels), 1)

class Game:
    __slots__ = ['winner', 'welo', 'belo', 'tc', 'white', 'black']
    def __init__(self, winner, welo, belo, tc, white = "", black = ""):
        self.winner = winner
        self.welo = welo
        self.belo = belo
        self.tc = tc
        self.white = white
        self.black = black


class Board:
    __slots__ = ['game', 'eval', 'fen']
    def __init__(self, game, fen, eval):
        self.game = game
        self.eval = eval
        self.fen = fen

    def __repr__(self):
        return self.fen

    def side_to_move(self):
        return 1.0 if self.fen.split(' ')[1] == "w" else -1.0

    def piece_indices(self):
        index = 0
        indices = []
        for char in self.fen.split(' ')[0]:
            if (char == '/'): continue
            if (char.isnumeric()): index += int(char)
            elif (char in ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']):
                indices.append(index + offsets[char])
                index += 1
            else: index += 1
        return indices

    def to_planes(self):
        to_move = self.side_to_move()
        planes = self.fen_to_planes()
        planes = torch.cat((planes, elo_to_plane(self.game.welo if to_move == 1.0 else self.game.belo)))
        planes = torch.cat((planes, tc_to_plane(self.game.tc)))
        planes = torch.cat((planes, eval_to_plane(self.eval)))
        return planes

    def fen_to_planes(self):
        planes = self.populate_piece_planes()
        to_move = torch.full((1, 8, 8), self.side_to_move())
        planes = torch.cat((planes, to_move))
        return planes

    def populate_piece_planes(self) -> torch.Tensor:
        planes = torch.zeros(12*64)
        new_indices = self.piece_indices()
        planes[new_indices] = 1.0
        planes = planes.reshape(12, 8, 8)
        return planes


def elo_to_plane(elo):
    AVG = 1500 # Approximate values, worth taking another
    STDDEV = 350 # look at once a baseline is established
    return torch.full((1, 8, 8), (elo-AVG)/STDDEV)

def tc_to_plane(tc):
    AVG = 800 # VERY approximate values, definitely worth taking another
    STDDEV = 200 # look at once a baseline is established
    return torch.full((1, 8, 8), (tc-AVG)/STDDEV)

def eval_to_plane(eval):
    return torch.full((1, 8, 8), eval) # check if passing next instead of current...

def eval_delta(eval, next_eval, next_to_move):
    return torch.tensor([(eval-next_eval)*next_to_move])
