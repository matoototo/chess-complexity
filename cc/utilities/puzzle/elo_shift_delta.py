import torch
from torch.utils.data.dataloader import DataLoader
from cc.parser.data import InferDataset, Board, Game
import json, argparse, pathlib

def shift(net, path, jump = 200, tc = 1200, target=0.15):
    positions = json.load(open(path, 'r+'))
    dataset_arr = []
    for position in positions:
        if position['elo'] in [470, 3020]: continue
        g = Game(0, position['elo']+jump, position['elo']+jump, tc)
        b = Board(g, position['fen'], position['eval'])
        dataset_arr.append(b)
    dataset = InferDataset(dataset_arr)
    with torch.no_grad():
        net.eval()
        net.to('cuda:0')
        shifted = []
        dataloader = DataLoader(dataset, 256)
        for planes, indices in dataloader:
            planes = planes.to('cuda:0')
            out = net(planes)
            for i, pred in zip(indices, out):
                shifted.append({ 'fen': dataset.positions[i].fen, 'elo': dataset.positions[i].game.welo,
                                 'eval': dataset.positions[i].eval, 'shift': target-pred.item(), 'pred': pred.item() })
    return shifted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate output delta when input Elo is shifted by a fixed amount.')
    parser.add_argument('-i', '--input', metavar='path', type=pathlib.Path, help='the path to the input json file', required=True)
    parser.add_argument('-o', '--output', metavar='path', type=pathlib.Path, help='the path to the output json file', required=True)
    parser.add_argument('--net', metavar='path', type=pathlib.Path, help='the network weights used (should be the same as the one used to assign Elo)', required=True)
    parser.add_argument('-j', '--jump', metavar='value', type=int, help='the difference in Elo', required=False, default=200)
    parser.add_argument('--target', metavar='value', type=float, help='the target used to assign the Elo', required=False, default=0.15)
    parser.add_argument('--tc', metavar='value', type=float, help='the time control used to assign the Elo', required=False, default=1200.0)

    args = parser.parse_args()

    from cc.utilities.evaluator import load_net
    assigned = shift(load_net(args.net), args.input, args.jump, args.tc, args.target)
    assigned.sort(key = lambda x : x['shift'])
    json.dump(assigned, open(args.output, 'w+'), indent=4)
