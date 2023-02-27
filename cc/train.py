from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from cc.model.model import Model
from cc.parser.data import PositionDataset
import cc.checkpoint as checkpoint
import torch
import torch.optim
import torch.utils.data
import torch.nn.utils
import torch.nn
import torch.linalg

import os
import argparse
import pathlib
import random

import yaml

from torchinfo import summary

def dir_to_data(dir):
    files = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith('.data'):
                files.append(os.path.join(root, filename))
    return files

parser = argparse.ArgumentParser(description='Train a network for complexity prediction')
parser.add_argument('--run', metavar='run number', type=int, help='the number of the training run, ex. 12', required=True)
parser.add_argument('--yaml', metavar='path', type=pathlib.Path, help='the path to the yaml file', required=True)
parser.add_argument('--empty-used', help='if present, empties the list of used files (usually done at end of epoch)', action='store_true')
parser.add_argument('--summary', help='if present, prints a summary of the model', action='store_true')

args = parser.parse_args()
config = yaml.load(open(args.yaml).read(), Loader=yaml.FullLoader)
data_c = config['data']
train_c = config['train']
model_c = config['model']
if 'head_v2' not in model_c: model_c['head_v2'] = False
if 'head_filters' not in model_c: model_c['head_filters'] = 1
if 'use_se' not in model_c: model_c['use_se'] = False
if 'se_ratio' not in model_c: model_c['se_ratio'] = 8
if 'warmup_steps' not in train_c: train_c['warmup_steps'] = 0
if 'block_activation' not in model_c: model_c['block_activation'] = 'ReLU'

train_c["val_size"] //= train_c["num_workers"]

run_number = args.run
data_base = os.path.abspath(data_c['db_dir'])
checkpoint_base = os.path.abspath(data_c['cp_dir'])
log_base = os.path.abspath(data_c['log_dir'])
val_base = os.path.abspath(data_c['val_dir'])
empty_used = args.empty_used

files = dir_to_data(data_base)
val_files = dir_to_data(val_base)

random.seed(run_number)
random.shuffle(files)
random.shuffle(val_files)

run_dir = f"run{int(run_number):2d}"
checkpoint_path = os.path.join(checkpoint_base, run_dir)
log_path = os.path.join(log_base, run_dir)

for p in [checkpoint_path, log_path]:
    if not os.path.exists(p): os.makedirs(p)


loss_func = torch.nn.MSELoss()
writer = SummaryWriter(log_path, purge_step=1000000)


def val(val_loader : DataLoader, net : torch.nn.Module):
    net.eval()
    val_loss = 0
    for x, y in val_loader:
        with torch.no_grad():
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            preds = net(x)
            val_loss += len(x) * loss_func(preds, y)
    return val_loss / (train_c["num_workers"]*len(val_loader.dataset))


def train(x, y, net : torch.nn.Module):
    net.train()
    x = x.to('cuda:0')
    y = y.to('cuda:0')
    optim.zero_grad()
    preds = net(x)
    loss = loss_func(preds, y)
    loss.backward()
    return loss


def load_optim(opt, net):
    opt_map = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RAdam': torch.optim.RAdam}
    if opt == 'SGD': optim = opt_map[opt](net.parameters(), train_c['lr'], 0.9)
    else: optim = opt_map[opt](net.parameters(), train_c['lr'])
    return optim


def linear_warmup(optim):
    if steps < train_c['warmup_steps']:
        for g in optim.param_groups:
            g['lr'] = (1+steps)/train_c['warmup_steps'] * train_c['lr']

def mask_to_list(mask, files):
    return [f for f, m in zip(files, mask) if m]

def list_to_mask(files, used):
    return [f in used for f in files]

activation_map = {'ReLU': torch.nn.ReLU, 'Mish': torch.nn.Mish}
block_activation = activation_map[model_c['block_activation']]

checkpoints = os.listdir(checkpoint_path)
if len(checkpoints) != 0:
    paths = [os.path.join(checkpoint_path, basename) for basename in checkpoints]
    newest_checkpoint = max(paths, key=os.path.getctime)
    cpnt = checkpoint.load(newest_checkpoint)

    steps = cpnt['steps']
    net = Model(*cpnt['model_args']).to('cuda:0')
    net.load_state_dict(cpnt['model_state'])
    optim = load_optim(train_c['optim'], net)
    optim.load_state_dict(cpnt['optim_state'])

    for g in optim.param_groups:
        g['lr'] = train_c['lr']

    used = cpnt['used_files']
    if empty_used: used = []
    files = list(filter(lambda x : x not in used, files))
else:
    net = Model(model_c['filters'], model_c['blocks'], model_c['head'], model_c['head_v2'],
                model_c['use_se'], model_c['se_ratio'], model_c['head_filters'], block_activation).to('cuda:0')
    net.reset_parameters()
    optim = load_optim(train_c['optim'], net)
    used = []
    steps = 0

if args.summary: summary(net, verbose=2)

val_every = train_c['val_every']
train_loss = 0

from multiprocessing.managers import SharedMemoryManager

smm = SharedMemoryManager()
smm.start()

used_mask = smm.ShareableList(list_to_mask(files, used))

val_dataset = PositionDataset(val_files, train_c["val_size"], loop=True)
val_loader = DataLoader(val_dataset, train_c['bs'], False, pin_memory=True, num_workers=train_c['num_workers'], persistent_workers=True)

train_dataset = PositionDataset(files, used = used_mask)
loader = DataLoader(train_dataset, train_c['bs'], pin_memory=True, drop_last=True, num_workers=train_c['num_workers'])
n_samples = 0
for x, y in loader:
    n_samples += len(x)
    linear_warmup(optim)
    train_loss += len(x) * train(x, y, net)
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), train_c['grad_norm'])
    optim.step()
    steps += 1
    if (steps%val_every == 0):
        val_loss = val(val_loader, net)
        train_scalar = train_loss/n_samples
        n_samples = 0

        print(f"Files used so far: {sum(used_mask)}")
        print(steps, ':', val_loss)

        writer.add_scalar("Loss/val", val_loss, steps)
        writer.add_scalar("Gradient norm/norm", norm, steps)
        writer.add_scalar("Loss/train", train_scalar, steps)
        writer.add_scalar("Learning rate/lr", optim.param_groups[0]['lr'], steps)
        for name, param in net.named_parameters():
            if param.requires_grad:
                writer.add_scalar(f"Weight norm/{name}", torch.linalg.norm(param), steps)
        flat_param = torch.nn.utils.parameters_to_vector(net.parameters())
        writer.add_scalar("Weight norm/reg term", flat_param.dot(flat_param), steps)
        writer.flush()

        train_loss = 0
        if (steps % (val_every * 4) == 0):
            checkpoint.save(
                steps,
                net.state_dict(),
                optim.state_dict(),
                mask_to_list(used_mask, files), net.args,
                os.path.join(checkpoint_path, f"{steps}.pt")
            )
checkpoint.save(
    steps,
    net.state_dict(),
    optim.state_dict(),
    mask_to_list(used_mask, files), net.args,
    os.path.join(checkpoint_path, f"{steps}.pt")
)
writer.close()
