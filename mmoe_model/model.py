import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
from data_process import train_loader, test_loader
import argparse
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.autograd import Variable

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
    t1_pred, t2_pred, t3_pred, t1_target, t2_target, t3_target = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        epoch_loss = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]
            yhat_1, yhat_2, yhat_3 = yhat[0], yhat[1], yhat[2]

            loss1 = bce_loss_fn(yhat_1, y1.view(-1, 1))
            loss2 = 0.000001 * mse_loss_fn(yhat_2, y2.view(-1, 1))
            loss3 = 0.00001 * mse_loss_fn(yhat_3, y3.view(-1, 1))
            loss = loss1 + loss2 + loss3

            t1_hat, t2_hat, t3_hat = list(yhat_1.cpu()), list(yhat_2.cpu()), list(yhat_3.cpu())

            t1_pred += t1_hat
            t2_pred += t2_hat
            t3_pred += t3_hat
            t1_target += list(y1.cpu())
            t2_target += list(y2.cpu())
            t3_target += list(y3.cpu())

    # t1_pred = [1 if x else 0 for x in list(t1_pred)]
    # t2_pred = [1 if x else 0 for x in list(t2_pred)]

    auc_1 = roc_auc_score(t1_target, t1_pred)
    # auc_2 = roc_auc_score(t2_target, t2_pred)
    # auc_3 = roc_auc_score(t3_target, t3_pred)
    # return auc_1, auc_2, auc_3
    

    mse_revenue = mean_squared_error(y_true=t2_target, y_pred=t2_pred)
    r2_revenue = r2_score(y_true=t2_target, y_pred=t2_pred)

    mse_reputation = mean_squared_error(y_true=t3_target, y_pred=t3_pred)
    r2_reputation = r2_score(y_true=t3_target, y_pred=t3_pred)
    
    return auc_1, mse_revenue, r2_revenue, mse_reputation, r2_reputation


model = MMOE(input_size=320, num_experts=6, experts_out=16, experts_hidden=32, towers_hidden=8, tasks=3)
model = model.to(device)
lr = 1e-4
bce_loss_fn = nn.BCELoss(reduction='mean')
mse_loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
val_loss = []
# 添加输入和预测记录的 list
inputs_list = []
preds_list = []

from torch.autograd import Variable
def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def train_model(n_epochs, train_loader, use_y1, use_y2, use_y3, report_train):
    losses = []
    initial_losses = [0,0,0] # Initial losses for y1, y2, y3
    alpha = 0.16 # GradNorm hyper-parameter

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        print("Epoch: {}/{}".format(epoch, n_epochs))
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            # 将输入信息记录下来
            inputs_list.append(x.cpu().numpy())

            # 将预测信息记录下来
            preds_list.append(y_hat.detach().cpu().numpy())

            y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]
            y_1, y_2, y_3 = y_hat[0], y_hat[1], y_hat[2]

            prev_loss1 = bce_loss_fn(y_1, y1.view(-1, 1)) if use_y1 else 0
            prev_loss2 = 0.000001 * mse_loss_fn(y_2, y2.view(-1, 1)) if use_y2 else 0
            prev_loss3 = 0.00001 * mse_loss_fn(y_3, y3.view(-1, 1)) if use_y3 else 0
            losses = [prev_loss1, prev_loss2, prev_loss3]
            weights = [Variable(loss / initial_loss if initial_loss != 0 else loss, requires_grad=True) for loss, initial_loss in zip(losses, initial_losses)]
            grad_norms = []
            for ind, task_loss in enumerate(weights):
                task_loss.backward(retain_graph=True)
                grad_norms.append(compute_grad_norm(model))
                model.zero_grad()

            avg_grad_norm = sum(grad_norms) / len(grad_norms)

            weight_loss = [weights[i]*losses[i] for i in range(len(weights))]

            loss = sum(weight_loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # GradNorm weight adjustment and normalization
            for i in range(len(weights)):
                weights[i] = weights[i] * (1 + alpha * (grad_norms[i] / avg_grad_norm - 1))

        epoch_loss.append(loss.item())

        losses.append(np.mean(epoch_loss))
        if report_train:
            auc1, mse_revenue, r2_revenue, mse_reputation, r2_reputation = test(train_loader)
        else:
            auc1, mse_revenue, r2_revenue, mse_reputation, r2_reputation = test(test_loader)
        
        print('train loss: {:.5f}, val task1 auc: {:.5f}, mse_revenue: {:.5f}, r2_revenue: {:.5f}, mse_reputation: {:.5f}, r2_reputation: {:.5f}'.format(np.mean(epoch_loss), auc1, mse_revenue, r2_revenue, mse_reputation, r2_reputation))


    # 训练模型结束后，将记录的输入和预测输出拼接起来
    inputs_arr = np.concatenate(inputs_list, axis=0)
    preds_arr = np.concatenate(preds_list, axis=0)

    # 创建 DataFrame
    df = pd.DataFrame(np.hstack([inputs_arr, preds_arr]))
    df.columns = ["x1", "x2", ..., "y_predicted1", "y_predicted2", ...] 

    # 保存为 csv 文件
    df.to_csv("result.csv", index=False)
def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument('--use_y1', action='store_true', help='Use y1 in loss calculation')
    parser.add_argument('--use_y2', action='store_true', help='Use y2 in loss calculation')
    parser.add_argument('--use_y3', action='store_true', help='Use y3 in loss calculation')
    parser.add_argument('--report_train', action='store_true', help='Report metric in train dataset')


    args = parser.parse_args()

    # Assume train_loader is predefined or loaded elsewhere
    train_model(args.epochs, train_loader, args.use_y1, args.use_y2, args.use_y3, args.report_train)

if __name__ == '__main__':
    main()
# auc1, auc2, auc3 = test(test_loader)
# print('test task1 auc: {:.3f}, test task2 auc: {:.3f}, test task3 auc: {:.3f}'.format(auc1, auc2, auc3))