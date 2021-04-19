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

checkpoint_path = "./checkpoints/run4/"

path = "/mnt/melem/Linux-data/chess-complex/fens/"
files = os.listdir(path)

loss_func = torch.nn.MSELoss()
writer = SummaryWriter("./logs/run4/", purge_step=9186)


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
if len(checkpoints) != 0:
    paths = [os.path.join(checkpoint_path, basename) for basename in checkpoints]
    newest_checkpoint = max(paths, key=os.path.getctime)
    cpnt = checkpoint.load(newest_checkpoint)

    steps = cpnt['steps']
    net = Model(*cpnt['model_args']).to('cuda:0')
    net.load_state_dict(cpnt['model_state'])

    optim = torch.optim.SGD(net.parameters(), 0.01, 0.9, nesterov=True, weight_decay=1e-4)
    optim.load_state_dict(cpnt['optim_state'])

    used = cpnt['used_files']
    files = list(filter(lambda x : x not in used, files))
else:
    net = Model(64, 6, 128).to('cuda:0')
    optim = torch.optim.SGD(net.parameters(), 0.01, 0.9, nesterov=True, weight_decay=1e-4)
    used = []
    steps = 0


test_dataset = PositionDataset("processed_0.data")
test_dataset.parse_data(100000)
test_loader = DataLoader(test_dataset, 1024, False, pin_memory=True)


for file in files:
    print(file)
    used.append(file)
    train_dataset = PositionDataset(path + file)
    if file == "processed_0.data":
        train_dataset.file.seek(test_dataset.file.tell())
    train_loss = 0
    while True:
        train_dataset.parse_data(100000)
        if (len(train_dataset.data) == 0): break
        loader = DataLoader(train_dataset, 1024, True, pin_memory=True)
        for x, y in loader:
            train_loss += train(x, y, net)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            steps += 1
            if (steps%100 == 0):
                test_loss = test(test_loader, net)
                print(steps, ':', test_loss)
                writer.add_scalar("Loss/test", test_loss, steps)
                writer.add_scalar("Gradient norm/norm", norm, steps)
                writer.add_scalar("Loss/train", train_loss/100, steps)
                train_loss = 0
    checkpoint.save(steps, net.state_dict(), optim.state_dict(), used, net.args, checkpoint_path + f"{steps}.pt")
writer.close()
