from torch.utils.data.dataloader import DataLoader
from model.model import Model, ComplexityHead
from parser.data import PositionDataset
import torch
import torch.optim
import torch.utils.data
import torch.nn


net = Model(16, 4, 128).to('cuda:0')
optim = torch.optim.SGD(net.parameters(), 0.01, 0.9, weight_decay=1e-5)
loss_func = torch.nn.MSELoss()



test_dataset = PositionDataset("processed_0.data")
test_dataset.parse_data(100000)
test_loader = DataLoader(test_dataset, 1024, False, pin_memory=True)


train_dataset = PositionDataset("processed_0.data")
train_dataset.file.seek(test_dataset.file.tell())

print(len(test_dataset.data))

for epoch in range(100):
    print("LOADING")
    train_dataset.parse_data(100000)
    loader = DataLoader(train_dataset, 1024, True, pin_memory=True)
    print("OK")
    for x, y in loader:
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        optim.zero_grad()
        preds = net(x)
        loss = loss_func(preds, y)
        loss.backward()
        optim.step()
    with torch.no_grad():
        test_loss = 0
        for x, y in test_loader:
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            preds = net(x)
            test_loss += loss_func(preds, y)
        print(epoch, ':', test_loss/len(test_loader))
