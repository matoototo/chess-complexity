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


def get_step_number(checkpoint_filename):
    return int(cp.split('.')[0])


def get_newest_checkpoint(path):
    checkpoints = [os.path.join(path, f) for f in os.listdir(path)]
    return max(checkpoints, key=os.path.getctime) if len(checkpoints) > 0 else None


parser = argparse.ArgumentParser(description='Train a network for complexity prediction')
parser.add_argument('--run', metavar='run number', type=str, help='the number of the training run, ex. 12', required=True)
parser.add_argument('--db', metavar='path', type=pathlib.Path, help='path to the dir containing .data training files', required=True)
parser.add_argument('--cp', metavar='path', type=pathlib.Path, help='path to the root dir of the checkpoints folders', required=True)
parser.add_argument('--log', metavar='path', type=pathlib.Path, help='path to the root dir of the logs folders', required=True)
parser.add_argument('--test', metavar='filepath', type=pathlib.Path, help='path to the testing .data file', required=False, default='shuffled_0.data')
parser.add_argument('--branch', metavar='step number', type=int, help='branch at checkpoint given by filename by creating <run>[a..z]', required=False)

args = parser.parse_args()

run_number = args.run
data_base = os.path.abspath(args.db)
checkpoint_base = os.path.abspath(args.cp)
log_base = os.path.abspath(args.log)
test_dataset_filename = args.test
branch = args.branch

files = os.listdir(data_base)

run_dir = f"run{run_number}"
checkpoint_path = os.path.join(checkpoint_base, run_dir)

used_runs = os.listdir(checkpoint_base)

if branch: # and branch != newest_checkpoint:
    extension = 'a'
    while (run_dir + extension) in used_runs:
        extension = chr(ord(extension) + 1)
    run_dir_master = run_dir
    run_dir += extension
    checkpoint_path_master = os.path.join(checkpoint_base, run_dir_master)
    checkpoint_path = os.path.join(checkpoint_base, run_dir)
    print(f"branching to {run_dir}")

log_path = os.path.join(log_base, run_dir)
for p in [checkpoint_path, log_path]:
    if not os.path.exists(p): os.makedirs(p)

if branch:
    from shutil import copyfile

    cp = f"{branch}.pt"
    copyfile(os.path.join(checkpoint_path_master, cp), os.path.join(checkpoint_path, cp))

    log_path_master = os.path.join(log_base, run_dir_master)
    for log in os.listdir(log_path_master):
        copyfile(os.path.join(log_path_master, log), os.path.join(log_path, log))

newest_checkpoint = get_newest_checkpoint(checkpoint_path)
purge_step = get_step_number(newest_checkpoint)

loss_func = torch.nn.MSELoss()
writer = SummaryWriter(log_path, purge_step=purge_step)

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


if newest_checkpoint:
    cpnt = checkpoint.load(newest_checkpoint)
    steps = cpnt['steps']
    net = Model(*cpnt['model_args']).to('cuda:0')
    net.load_state_dict(cpnt['model_state'])

    optim = torch.optim.SGD(net.parameters(), 3e-4)
    optim.load_state_dict(cpnt['optim_state'])

    used = cpnt['used_files']
    files = list(filter(lambda x : x not in used, files))
else:
    net = Model(128, 10, 128).to('cuda:0')
    net.reset_parameters()
    optim = torch.optim.SGD(net.parameters(), 3e-4)
    used = []
    steps = 0

test_dataset = PositionDataset(os.path.join(data_base, test_dataset_filename))
test_dataset.parse_data(100000)
test_loader = DataLoader(test_dataset, 1024, False, pin_memory=True, num_workers=2)

test_every = 500
train_loss = 0
for file in files:
    if file.split('.')[-1] != 'data': continue
    print(file)
    used.append(file)
    train_dataset = PositionDataset(os.path.join(data_base, file))
    if file == test_dataset_filename:
        train_dataset.file.seek(test_dataset.file.tell())
    while True:
        train_dataset.parse_data()
        if (len(train_dataset) == 0): break
        if (len(train_dataset) < 100000):
            print("throwing!", len(train_dataset))
            break
        loader = DataLoader(train_dataset, 1024, True, pin_memory=True, drop_last=True, num_workers=2)
        for x, y in loader:
            train_loss += train(x, y, net)
            # norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 4.0)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 4.0)
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
