import torch
from cc.parser.data import PositionDataset, tc_to_plane, elo_to_plane
from cc.utilities.evaluator import load_net
import argparse, pathlib, json

EPS = 0.05

def i_to_elo(i, min = 470.0, step = 10.0):
    return min + i*step

def naive_assign(net, dataset : PositionDataset, target = 0.2, tc = 600):
    # assumes Elo range of 470 - 3030, 5~ Elo acc
    with torch.no_grad():
        net.eval()
        net.to('cuda:0')
        assigned = []
        for i in range(len(dataset)):
            if not i%100: print(i)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign Elo to positions.')
    parser.add_argument('-i', '--input', metavar='path', type=pathlib.Path, help='the path to the input .data file', required=True)
    parser.add_argument('-o', '--output', metavar='path', type=pathlib.Path, help='the path to the output .json file', required=True)
    parser.add_argument('--net', metavar='path', type=pathlib.Path, help='the network weights used to assign Elo', required=True)
    parser.add_argument('--target', metavar='value', type=float, help='the target used to assign the Elo', required=False, default=0.15)
    parser.add_argument('--tc', metavar='value', type=float, help='the time control used to assign the Elo', required=False, default=1200.0)
    parser.add_argument('-l', '--limit', metavar='value', type=int, help='only assign Elo to the first l', required=False)

    args = parser.parse_args()

    dataset = PositionDataset(args.input)
    dataset.parse_data(args.limit)

    assigned = naive_assign(load_net(args.net), dataset, args.target, args.tc)
    assigned.sort(key = lambda x : x['elo'])
    json.dump(assigned, open(args.output, 'w+'), indent=4)
