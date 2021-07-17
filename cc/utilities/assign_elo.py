import torch
from cc.parser.data import PositionDataset, tc_to_plane, elo_to_plane

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
            assigned.append({ 'fen': dataset.positions[i].fen, 'elo': i_to_elo(min), 'eval': dataset.positions[i].eval})
    return assigned

