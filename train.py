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
import sys

run_number = sys.argv[1]
data_base = os.path.abspath(sys.argv[2])
checkpoint_base = os.path.abspath(sys.argv[3])
log_base = os.path.abspath(sys.argv[4])

if os.path.splitext(os.path.splitext(data_base)[0])[1] == ".tar":
  import tarfile
  MODE_TAR = True
  data = tarfile.open(data_base, 'r:xz')
  data_base = os.path.dirname(data_base)
  files = data.getmembers()
else:
  MODE_TAR = False
  files = os.listdir(data_base)
print(files)

run_dir = f"run{int(run_number):2d}"
checkpoint_path = os.path.join(checkpoint_base, run_dir)
log_path = os.path.join(log_base, run_dir)

for p in [checkpoint_path, log_path]:
  if not os.path.exists(p): os.makedirs(p)

test_dataset_filename = "shuffled_0.data"

loss_func = torch.nn.MSELoss()
writer = SummaryWriter(log_path, purge_step=287013)

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

    optim = torch.optim.Adam(net.parameters(), 3e-4)
    optim.load_state_dict(cpnt['optim_state'])

    used = cpnt['used_files']
    files = list(filter(lambda x : x not in used, files))
else:
    net = Model(128, 10, 128).to('cuda:0')
    net.reset_parameters()
    optim = torch.optim.Adam(net.parameters(), 3e-4)
    used = []
    steps = 0
  
test_dataset = PositionDataset(os.path.join(data_base, test_dataset_filename))
test_dataset.parse_data(100000)
test_loader = DataLoader(test_dataset, 1024, False, pin_memory=True, num_workers=2)

test_every = 500
train_loss = 0
for file in files:
    if file in ["0-100.tar.xz", ".ipynb_checkpoints"]: continue
    if MODE_TAR: 
      data.extractall(data_base, [file])
      file = file.name
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
    if MODE_TAR: os.remove(os.path.join(data_base, file))
writer.close()
