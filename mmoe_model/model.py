import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


random.seed(3)
np.random.seed(3)
seed = 3
batch_size = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class MMOE(nn.Module):
    def __init__(self, input_size, num_experts, experts_out, experts_hidden, towers_hidden, tasks):
        super(MMOE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(x @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return final_output


def test(loader):
    t1_pred, t2_pred, t1_target, t2_target = [], [], [], []
    model.eval()
    with torch.no_grad():
        epoch_loss = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            y1, y2 = y[:, 0],y[:, 1]
            yhat_1, yhat_2 = yhat[0], yhat[1]

            loss1, loss2 = loss_fn(yhat_1, y1.view(-1, 1)), loss_fn(yhat_2, y2.view(-1, 1))
            loss = loss1 + loss2

            # t1_hat = yhat_1.view(-1) > 0.7
            # t2_hat = yhat_2.view(-1) > 0.5
            t1_hat, t2_hat = list(yhat_1.cpu()), list(yhat_2.cpu())

            t1_pred += t1_hat
            t2_pred += t2_hat
            t1_target += list(y1.cpu())
            t2_target += list(y2.cpu())

    # t1_pred = [1 if x else 0 for x in list(t1_pred)]
    # t2_pred = [1 if x else 0 for x in list(t2_pred)]

    auc_1 = roc_auc_score(t1_target, t1_pred)
    auc_2 = roc_auc_score(t2_target, t2_pred)
    return auc_1, auc_2


model = MMOE(input_size=320, num_experts=6, experts_out=16, experts_hidden=32, towers_hidden=8, tasks=3)
model = model.to(device)
lr = 1e-4
n_epochs = 10
loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
losses = []
val_loss = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = []
    c = 0
    print("Epoch: {}/{}".format(epoch, n_epochs))
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        y1, y2 = y[:, 0], y[:, 1]
        y_1, y_2 = y_hat[0], y_hat[1]

        loss1 = loss_fn(y_1, y1.view(-1, 1))
        loss2 = loss_fn(y_2, y2.view(-1, 1))
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(loss.item())
    losses.append(np.mean(epoch_loss))

    auc1, auc2 = test(val_loader)
    print('train loss: {:.5f}, val task1 auc: {:.5f}, val task2 auc: {:.3f}'.format(np.mean(epoch_loss), auc1, auc2))

auc1, auc2 = test(test_loader)
print('test task1 auc: {:.3f}, test task2 auc: {:.3f}'.format(auc1, auc2))