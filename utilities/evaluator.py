import sys, os, io

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parser')))

import checkpoint
import model.model as model

from torch import no_grad
from torch.utils.data.dataloader import DataLoader

def load_net(path):
    """Loads the network Model to CPU from checkpoint at given path."""
    chkpt = checkpoint.load(path)
    net = model.Model(*chkpt["model_args"])
    net.load_state_dict(chkpt["model_state"])
    return net

def eval_data(net, dataset, bs = 128, shuffle = False):
    """Evaluates every position in the InferDataset and returns a List of Board-eval pairs."""
    with no_grad():
        net.eval()
        data_loader = DataLoader(dataset, bs, shuffle)
        out = []
        for pos, indices in data_loader:
            pos = pos.to('cuda:0')
            out += zip(net(pos).squeeze().tolist(), [dataset.positions[i] for i in indices])
    return out

