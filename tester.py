from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from model.model import Model, ComplexityHead
import parser.data
import checkpoint
import torch
import torch.optim
import torch.utils.data
import torch.nn.utils
import torch.nn
import os

# 2r1k2r/1p1bqppp/p2b4/n3p3/3pP3/3P1N2/PP1Q1PPP/R1B1KB1R w KQk - 0 15
# 2r1k2r/1p1b1p1p/pq3p2/n3p3/1b1pP3/1P1P1N2/P2QBPPP/R1B2RK1 b k - 4 17
# 6r1/1p1k1p2/p1n2p2/4pP1p/P2pP2P/1P4P1/5q2/2R1R2K b - - 1 38

cpnt = checkpoint.load("checkpoints/run9/891360.pt")
net = Model(*cpnt['model_args']).to('cuda:0')
net.load_state_dict(cpnt['model_state'])
net.eval()

while True:
    board = parser.data.Board(input("FEN: "))
    planes = board.to_planes()
    planes = planes.reshape((1, 15, 8, 8)).to('cuda:0')
    print(net(planes))
