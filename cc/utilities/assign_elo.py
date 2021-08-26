import torch
from cc.parser.data import PositionDataset, tc_to_plane, elo_to_plane
from cc.utilities.evaluator import load_net
from cc.utilities.pgn_to_data import transform_eval
import argparse, pathlib, json
import chess
import chess.engine

EPS = 0.05

def log(val, max, message):
    print('\r' + " "*50 + '\r', end="")
    print(message + str(val) + '/' + str(max) + '\r', end="")

def i_to_elo(i, min = 470.0, step = 10.0):
    return min + i*step

def naive_assign(net, dataset : PositionDataset, target = 0.2, tc = 600, quiet = False):
    # assumes Elo range of 470 - 3030, 5~ Elo acc
    with torch.no_grad():
        net.eval()
        net.to('cuda:0')
        assigned = []
        for i in range(len(dataset)):
            if not quiet: log(i, len(dataset), "Assign progress: ")
            planes, label = dataset[i]
            planes = planes
            planes[-2] = tc_to_plane(tc)
            planes = torch.reshape(planes, (1, 16, 8, 8))
            planes = planes.repeat(256, 1, 1, 1)
            for plane, j in zip(planes, range(256)):
                plane[-3] = elo_to_plane(i_to_elo(j))
            planes = planes.to('cuda:0')
            out = torch.max(net(planes), torch.tensor(0.05).to('cuda:0'))
            min = torch.argmin(torch.abs(out-target)).item()
            if out[min] >= target - EPS:
                assigned.append({ 'fen': dataset.positions[i].fen, 'elo': i_to_elo(min), 'eval': dataset.positions[i].eval})
    return assigned

def eval_all(engine: chess.engine.SimpleEngine, assigned, quiet = False):
    engine.configure({"Threads": 4})
    for i, position in enumerate(assigned):
        if not quiet: log(i, len(assigned), "Eval progress: ")
        res = engine.analyse(chess.Board(position['fen']), chess.engine.Limit(depth=20), multipv=10)
        res = [{'eval': transform_eval(m['score']), 'move': m['pv'][0].__str__()} for m in res]
        position['moves'] = res
    engine.quit()
    return assigned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign Elo to positions and evaluate with multipv=10 if the engine is specified.')
    parser.add_argument('-i', '--input', metavar='path', type=pathlib.Path, help='the path to the input .data file', required=True)
    parser.add_argument('-o', '--output', metavar='path', type=pathlib.Path, help='the path to the output .json file', required=True)
    parser.add_argument('--net', metavar='path', type=pathlib.Path, help='the network weights used to assign Elo', required=True)
    parser.add_argument('--engine', metavar='path', type=pathlib.Path, help='the path to the engine used for evaluation', required=False)
    parser.add_argument('--target', metavar='value', type=float, help='the target used to assign the Elo', required=False, default=0.15)
    parser.add_argument('--tc', metavar='value', type=float, help='the time control used to assign the Elo', required=False, default=1200.0)
    parser.add_argument('-l', '--limit', metavar='value', type=int, help='only assign Elo to the first l', required=False)
    parser.add_argument('-q', '--quiet', action='store_true', help='if present, doesn\'t print log messages.', default=False)

    args = parser.parse_args()

    dataset = PositionDataset(args.input)
    dataset.parse_data(args.limit)

    assigned = naive_assign(load_net(args.net), dataset, args.target, args.tc, args.quiet)
    if args.engine:
        assigned = eval_all(chess.engine.SimpleEngine.popen_uci(args.engine), assigned, args.quiet)

    assigned.sort(key = lambda x : x['elo'])
    json.dump(assigned, open(args.output, 'w+'), indent=4)
