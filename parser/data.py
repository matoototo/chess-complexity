import torch
import torch.utils.data
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader


class PositionDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename)
        self.data = []
        self.labels = []

    def __getitem__(self, i):
        return (self.data[i], self.labels[i])

    def __len__(self):
        return len(self.data)

    def parse_data(self, limit):
        # Parse the next _limit_ entries in the file
        self.data = []
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
            planes = None
            eval = None
            while (True):
                fen = self.file.readline()
                if (len(fen) < 5):
                    self.data.pop()
                    if (len(self.data) > limit):
                        return (self.data, self.labels)
                    break
                board = Board(fen)
                to_move = board.side_to_move()
                old_eval = eval
                eval = float(self.file.readline())
                planes = fen_to_planes(board)
                planes = torch.cat((planes, elo_to_plane(welo if to_move == 1.0 else belo)))
                self.data.append(planes)
                if old_eval != None: # isn't first pos
                    self.labels.append(eval_delta(old_eval, eval, to_move))
            # print('after: ', file.readline())
        return (self.data, self.labels)


class Game:
    def __init__(self, winner, welo, belo, tc):
        self.winner = winner
        self.welo = welo
        self.belo = belo
        self.tc = tc


class Board:
    def __init__(self, fen):
        self.fen = fen.split(" ")
        self.offset = self.create_offsets()

    def create_offsets(self):
        offset = defaultdict(int)
        for char, i in zip(['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P'], range(12)):
            offset[char] = i*64
        return offset

    def side_to_move(self):
        return 1.0 if self.fen[1] == "w" else -1.0

    def piece_indices(self):
        index = 0
        indices = []
        for char in self.fen[0]:
            if (char == '/'): continue
            if (char.isnumeric()): index += int(char)
            elif (char in ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']):
                indices.append(index + self.offset[char])
                index += 1
            else: index += 1
        return indices


def fen_to_planes(board):
    planes = populate_piece_planes(board)
    to_move = torch.full((1, 8, 8), board.side_to_move())
    planes = torch.cat((planes, to_move))
    return planes

def elo_to_plane(elo):
    AVG = 1500 # Approximate values, worth taking another
    STDDEV = 350 # look at once a baseline is established
    return torch.full((1, 8, 8), (elo-AVG)/STDDEV)

def eval_delta(eval, next_eval, to_move):
    return torch.tensor([(eval-next_eval)*to_move])

def populate_piece_planes(board : Board) -> torch.Tensor:
    planes = torch.zeros(12*64)
    new_indices = board.piece_indices()
    planes[new_indices] = 1.0
    planes = planes.reshape(12, 8, 8)
    return planes

