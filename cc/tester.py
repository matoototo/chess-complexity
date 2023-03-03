from cc import checkpoint
from cc.model import Model
from cc.parser import data

import chess
import chess.engine
import argparse
import pathlib

class Prediction:
    def __init__(self, pos, pred) -> None:
        self.pos = pos
        self.pred = pred

def fen_eval(fen, engine_path, depth = 20):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    engine.quit()
    return (2*info["score"].wdl(model="sf15.1").white().expectation())-1.0

def evaluate_data(data_path, engine_path, depth = 20, limit = None):
    preds_fens = []
    for i, line in enumerate(open(data_path, 'r')):
        if limit is not None and i >= limit: break
        if i % 10 == 0: print(i, flush=True)
        line = line.split(',')
        file_fen = line[0]
        file_to_move = 1.0 if file_fen.split(' ')[1] == 'w' else -1.0

        game = data.Game(file_to_move, elo, elo, tc)
        board = data.Board(game, file_fen, fen_eval(file_fen, engine_path, depth))
        planes = board.to_planes()
        planes = planes.reshape((1, 16, 8, 8)).to('cuda:0')
        pred = net(planes)
        preds_fens.append(Prediction(board, pred.item()))
    return preds_fens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_path", type=pathlib.Path)
    parser.add_argument("checkpoint_path", type=pathlib.Path)
    parser.add_argument("--data_path", type=pathlib.Path, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of positions to evaluate")
    parser.add_argument("--elo", type=int, default=1500, help="Elo of player-to-move")
    parser.add_argument("--tc", type=int, default=600, help="Time control considered")


    args = parser.parse_args()
    if args.data_path is None and not args.interactive:
        raise ValueError("Must specify data_path or interactive")

    cpnt = checkpoint.load(args.checkpoint_path)
    net = Model(*cpnt['model_args']).to('cuda:0')
    net.load_state_dict(cpnt['model_state'])
    net.eval()

    elo = args.elo
    tc = args.tc

    if args.interactive:
        while True:
            fen = input("FEN: ")
            elo = int(input("Elo: "))
            game = data.Game(1.0, elo, elo, 600)
            board = data.Board(game, fen, fen_eval(fen))
            planes = board.to_planes()
            planes = planes.reshape((1, 16, 8, 8)).to('cuda:0')
            print(net(planes))
    else:
        preds_fens = evaluate_data(args.data_path, args.engine_path, args.depth, args.limit)
        for pred_fen in sorted(preds_fens, key = lambda x : x.pred, reverse=False):
            print(pred_fen.pos.fen, pred_fen.pred)
