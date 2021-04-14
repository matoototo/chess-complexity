import torch
from collections import defaultdict

class Loader:
    def __init__(self, filename):
        self.filename = filename
        self.data, self.labels = self.parse_data()
    def parse_data(self):
        game = None
        data = []
        labels = []
        file = open(self.filename)
        while True:
            try: # Ugly way to catch the end of the file
                winner = int(file.readline())
            except:
                break
            welo = int(file.readline())
            belo = int(file.readline())
            tc = int(file.readline())
            planes = None
            eval = None
            while (True):
                fen = file.readline()
                if (len(fen) < 5):
                    data.pop()
                    print(len(data), len(labels))
                    # if (len(data) > 100000): exit()
                    break
                board = Board(fen)
                to_move = board.side_to_move()
                old_eval = eval
                eval = float(file.readline())
                planes = fen_to_planes(board)
                planes = torch.cat((planes, elo_to_plane(welo if to_move == 1.0 else belo)))
                data.append(planes)
                if old_eval != None: # isn't first pos
                    labels.append(eval_delta(old_eval, eval, to_move))
            # print('after: ', file.readline())
        print(len(data), len(labels))
        return (data, labels)
class Game:
    def __init__(self, winner, welo, belo, tc):
        self.winner = winner
        self.welo = welo
        self.belo = belo
        self.tc = tc

class Board:
    def __init__(self, fen):
        self.fen = fen.split(" ")
    def side_to_move(self):
        return 1.0 if self.fen[1] == "w" else -1.0
    def piece_indices(self):
        index = 0
        indices = defaultdict(list)
        for char in self.fen[0]:
            if (char == '/'): continue
            if (char.isnumeric()): index += int(char)
            elif (char in ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']):
                indices[char].append(index)
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
    return (eval-next_eval)*to_move

def populate_piece_planes(board) -> torch.Tensor:
    planes = torch.zeros(12, 8, 8)
    pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
    all_indices = board.piece_indices()
    for piece, i in zip(pieces, range(12)):
        indices = all_indices[piece]
        for index in indices:
            planes[i][index//8][index%8] = 1.0
    return planes

