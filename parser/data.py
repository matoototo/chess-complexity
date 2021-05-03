import torch
import torch.utils.data
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader


def create_offsets():
    offset = defaultdict(int)
    for char, i in zip(['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P'], range(12)):
        offset[char] = i*64
    return offset

offsets = create_offsets()


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
            try: # Ugly way to catch the end of the file
                winner = int(self.file.readline())
            except:
                # swap files here?
                break
            welo = int(self.file.readline())
            belo = int(self.file.readline())
            tc = int(self.file.readline())
            game = Game(winner, welo, belo, tc)
            eval = None
            while (True):
                fen = self.file.readline()
                if (len(fen) < 5):
                    self.positions.pop() # pop last since it has no label
                    if (limit and len(self.positions) >= limit): return
                    break
                old_eval = eval
                eval = float(self.file.readline())
                self.positions.append(Board(game, fen, eval))
                if old_eval != None: # isn't first pos
                    self.labels.append(eval_delta(old_eval, eval, self.positions[-1].side_to_move()))


class Game:
    def __init__(self, winner, welo, belo, tc):
        self.winner = winner
        self.welo = welo
        self.belo = belo
        self.tc = tc


class Board:
    def __init__(self, game, fen, eval):
        self.game = game
        self.eval = eval
        self.fen = fen

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
    return torch.tensor([(eval-next_eval)*next_to_move*-1])


