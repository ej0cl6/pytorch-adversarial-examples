import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def train(model, device, loader_tr, optimizer, n_epoch):
    model.train()
    for epoch in range(1, n_epoch+1):
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('epoch {}: {}/{}'.format(epoch, batch_idx*len(x), len(loader_tr.dataset)))

def predict(model, device, loader_te):
    model.eval()
    P = torch.zeros(len(loader_te.dataset), dtype=torch.int64)
    with torch.no_grad():
        for x, y, idxs in loader_te:
            x, y = x.to(device), y.to(device)
            out = model(x)
            p = out.max(1)[1]
            P[idxs] = p.cpu()

    return P
