from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from cc.model.model import Model, ComplexityHead
from cc.parser import data
from cc import checkpoint
import torch
import torch.optim
import torch.utils.data
import torch.nn.utils
import torch.nn
import os
import chess
import chess.engine


def fen_eval(fen, depth = 20):
    engine = chess.engine.SimpleEngine.popen_uci("/home/matoototo/Board games/Chess/Engines/SF13")
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    engine.quit()
    return (2*info["score"].wdl(model="lichess").white().expectation())-1.0

# 2r1k2r/1p1bqppp/p2b4/n3p3/3pP3/3P1N2/PP1Q1PPP/R1B1KB1R w KQk - 0 15
# 2r1k2r/1p1b1p1p/pq3p2/n3p3/1b1pP3/1P1P1N2/P2QBPPP/R1B2RK1 b k - 4 17
# 6r1/1p1k1p2/p1n2p2/4pP1p/P2pP2P/1P4P1/5q2/2R1R2K b - - 1 38

cpnt = checkpoint.load("checkpoints/run16/1366932.pt")
net = Model(*cpnt['model_args']).to('cuda:0')
net.load_state_dict(cpnt['model_state'])
# print(cpnt['used_files'])
net.eval()

# for p in net.named_parameters():
#     print(p)

class Prediction:
    def __init__(self, pos, pred) -> None:
        self.pos = pos
        self.pred = pred


dataset = parser.data.PositionDataset("/mnt/melem/Linux-data/chess-complex/fens/2021-04/2021-04_processed_0.data")
dataset.parse_data(10000)
outputs = []
for x in dataset.positions:
    game = parser.data.Game(1.0, 1500, 1500, 600)
    x.game = game
    outputs.append(Prediction(x, net(x.to_planes().reshape((1, 16, 8, 8)).to('cuda:0')).item()))

outputs = sorted(outputs, key = lambda x : x.pred, reverse=False)
for output in outputs:
    print(output.pos.fen, output.pred)


# while True:
#     fen = input("FEN: ")
#     elo = int(input("Elo: "))
#     game = parser.data.Game(1.0, elo, elo, 600)
#     board = parser.data.Board(game, fen, fen_eval(fen))
#     planes = board.to_planes()
#     planes = planes.reshape((1, 16, 8, 8)).to('cuda:0')
#     print(net(planes))
