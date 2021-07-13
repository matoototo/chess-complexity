from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from model.model import Model, ComplexityHead
from parser.data import PositionDataset
import checkpoint
import torch
import torch.optim
import torch.utils.data
import torch.nn.utils
import torch.nn

import os
import argparse
import pathlib

import yaml

parser = argparse.ArgumentParser(description='Train a network for complexity prediction')
parser.add_argument('--run', metavar='run number', type=int, help='the number of the training run, ex. 12', required=True)
parser.add_argument('--yaml', metavar='path', type=pathlib.Path, help='the path to the yaml file', required=True)

args = parser.parse_args()
config = yaml.load(open(args.yaml).read(), Loader=yaml.FullLoader)
data_c = config['data']
train_c = config['train']
model_c = config['model']
if 'head_v2' not in model_c: model_c['head_v2'] = False

run_number = args.run
data_base = os.path.abspath(data_c['db_dir'])
checkpoint_base = os.path.abspath(data_c['cp_dir'])
log_base = os.path.abspath(data_c['log_dir'])
test_dataset_file = os.path.abspath(data_c['test_file'])

files = os.listdir(data_base)

run_dir = f"run{int(run_number):2d}"
checkpoint_path = os.path.join(checkpoint_base, run_dir)
log_path = os.path.join(log_base, run_dir)

for p in [checkpoint_path, log_path]:
    if not os.path.exists(p): os.makedirs(p)


loss_func = torch.nn.MSELoss()
writer = SummaryWriter(log_path, purge_step=1728110)

def test(test_loader : DataLoader, net : torch.nn.Module):
    net.eval()
    test_loss = 0
    for x, y in test_loader:
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        preds = net(x)
        with torch.no_grad():
            test_loss += loss_func(preds, y)
    return test_loss/len(test_loader)


def train(x, y, net : torch.nn.Module):
    net.train()
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    optim.zero_grad()
    preds = net(x)
    loss = loss_func(preds, y)
    loss.backward()
    return loss


checkpoints = os.listdir(checkpoint_path)
opt_map = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}
if len(checkpoints) != 0:
    paths = [os.path.join(checkpoint_path, basename) for basename in checkpoints]
    newest_checkpoint = max(paths, key=os.path.getctime)
    cpnt = checkpoint.load(newest_checkpoint)

    steps = cpnt['steps']
    net = Model(*cpnt['model_args']).to('cuda:0')
    net.load_state_dict(cpnt['model_state'])
    optim = opt_map[train_c['optim']](net.parameters(), train_c['lr'])
    optim.load_state_dict(cpnt['optim_state'])

    used = cpnt['used_files']
    files = list(filter(lambda x : x not in used, files))
else:
    net = Model(model_c['filters'], model_c['blocks'], model_c['head'], model_c['head_v2']).to('cuda:0')
    net.reset_parameters()
    optim = opt_map[train_c['optim']](net.parameters(), train_c['lr'])
    used = []
    steps = 0

test_dataset = PositionDataset(test_dataset_file)
test_dataset.parse_data(train_c['test_size'])
test_loader = DataLoader(test_dataset, train_c['bs'], False, pin_memory=True, num_workers=train_c['num_workers'])

test_every = train_c['test_every']
train_loss = 0
for file in files:
    if file.split('.')[-1] != 'data': continue
    print(file)
    used.append(file)
    train_dataset = PositionDataset(os.path.join(data_base, file))
    if file == test_dataset_file.split('/')[-1]:
        print("skipping")
        train_dataset.file.seek(test_dataset.file.tell())
    while True:
        train_dataset.parse_data()
        if (len(train_dataset) == 0): break
        if (len(train_dataset) < 100000):
            print("throwing!", len(train_dataset))
            break
        loader = DataLoader(train_dataset, train_c['bs'], True, pin_memory=True, drop_last=True, num_workers=train_c['num_workers'])
        for x, y in loader:
            train_loss += train(x, y, net)
            # norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 4.0)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), train_c['grad_norm'])
            optim.step()
            steps += 1
            if (steps%test_every == 0):
                test_loss = test(test_loader, net)
                print(steps, ':', test_loss)
                writer.add_scalar("Loss/test", test_loss, steps)
                writer.add_scalar("Gradient norm/norm", norm, steps)
                writer.add_scalar("Loss/train", train_loss/test_every, steps)
                writer.flush()
                train_loss = 0
    checkpoint.save(
      steps,
      net.state_dict(),
      optim.state_dict(),
      used, net.args,
      os.path.join(checkpoint_path, f"{steps}.pt")
    )
writer.close()
