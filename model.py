import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torch.utils.data import DataLoader
from preprocess import FactorDataset, LSTMDataset, tensor_minmax
import matplotlib.pyplot as plt

class FactorModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, lower, upper):
        super(FactorModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=-0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.hardtanh = nn.Hardtanh(min_val=lower, max_val=upper)

        n = 4
        self.c = nn.Parameter(torch.tensor(0.1)) # 较大的 c 倾向于更分散的权重分布
        b = cp.Parameter(n, nonneg=True) # 风险预算，前向传播
        Q_sqrt = cp.Parameter((n, n)) # 斜方差矩阵的平方根
        y = cp.Variable(n)   

        obj = cp.Minimize(cp.sum_squares(Q_sqrt @ y)) # 最小化组合的方差，控制总风险

        cons = [
            y >= 0, 
            b.T @ cp.log(y) >= self.c.detach().numpy(), # 每个资产满足特定的风险分配，对数函数线性化一些非线性关系具有凸优化的特性
        ]

        prob = cp.Problem(obj, cons)
        self.cvxpy_layer = CvxpyLayer(
            prob, 
            parameters=[b, Q_sqrt], 
            variables=[y]
        )

    def forward(self, x, Q_sqrt):
        b = self.fc1(x.view(x.size(0), -1)) # [batch_size, 5, 11] -> [batch_size, 5*11]
        b = self.leaky_relu(b)
        b = self.fc2(b)
        b = self.softmax(b)
        b = self.hardtanh(b)
        b = tensor_minmax(b)

        y, = self.cvxpy_layer(b, Q_sqrt)
        w = y / y.sum(dim=1, keepdim=True)
        return w
    
        # b = self.fc1(x)
        # b = self.leaky_relu(b)
        # b = self.fc2(b)
        # b = self.softmax(b)
        # b = self.hardtanh(b)
        # b = tensor_minmax(b)
        # b = b.view(b.size(0), -1)

        # y, = self.cvxpy_layer(b, Q_sqrt)
        # w = y / y.sum(dim=1, keepdim=True)
        # return w

class LSTMModel(nn.Module):
    
    def __init__(self, input_dim, lstm_hidden_dim, fc_hidden_dim, output_dim, lower, upper): 
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc_hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=-0.1)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.hardtanh = nn.Hardtanh(min_val=lower, max_val=upper)
        
        n = output_dim
        self.c = nn.Parameter(torch.tensor(0.1))  # 风险分散系数
        
        b = cp.Parameter(n, nonneg=True)
        Q_sqrt = cp.Parameter((n, n))
        y = cp.Variable(n)

        obj = cp.Minimize(cp.sum_squares(Q_sqrt @ y))  

        cons = [
            y >= 0,
            b.T @ cp.log(y) >= self.c.detach().numpy(),  
        ]

        prob = cp.Problem(obj, cons)
        self.cvxpy_layer = CvxpyLayer(prob, parameters=[b, Q_sqrt], variables=[y])

    def forward(self, x, Q_sqrt):
        # LSTM 前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)  # 输入 x 大小为 [batch_size, time_step, lstm_input_dim] 过去 30 个交易日 5 个资产的日收益率
        last_output = lstm_out[:, -1, :] # 最后一个时间步长的隐藏层

        # 全连接层前向传播
        b = self.fc1(last_output)
        b = self.leaky_relu(b)
        b = self.fc2(b)
        b = self.softmax(b)
        b = self.hardtanh(b)
        b = tensor_minmax(b)
        
        # 风险预算层
        y, = self.cvxpy_layer(b, Q_sqrt)
        w = y / y.sum(dim=1, keepdim=True)
        return w

       
def train_model(train_dataloader, val_dataloader, model, optimizer, epochs=50, early_stopping=10):
    patience_counter = 0
    best_loss = np.inf
    train_losses = []
    val_losses = []
    best_model = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for idx, (batch_features, batch_Q_sqrt, batch_labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            weights = model(batch_features, batch_Q_sqrt)
            ret = torch.mul(weights, batch_labels)
            loss = -torch.sum(ret)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if idx % 5 == 0:
            #     print('Current epoch: %d, Current batch: %d, Loss is %.3f' %(epoch+1,idx+1,loss.item()))

        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)

        val_loss = test_model(val_dataloader, model)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss}, Validation Loss: {val_loss}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping:
            print("Early stopping")
            break
    return train_losses, val_losses, best_model

def test_model(dataloader, model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for idx, (batch_features, batch_Q_sqrt, batch_labels) in enumerate(dataloader):
            weights = model(batch_features, batch_Q_sqrt)
            ret = torch.mul(weights, batch_labels)
            loss = -torch.sum(ret)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def rolling_train(data, window_size=1800, train_size=1500, val_size=300, roll_step=100, gap=21, epochs=50, early_stopping=10):
    data_slice = 29 # 29 for LSTM, 0 for Factor
    levels = data.index.get_level_values(0).drop_duplicates()
    data = data.loc[levels]
    max_index = len(levels) - window_size - gap * 2 - roll_step - data_slice
    i = 0
    predictions = pd.DataFrame()

    while i <= max_index:
        print(f"Rolling Window: {i}/{max_index}")
        
        window_dates = levels[i:i + window_size + gap * 2 + roll_step + data_slice]
        train_dates = window_dates[:train_size + data_slice]
        val_dates = window_dates[train_size + gap:train_size + gap + val_size + data_slice]
        test_dates = window_dates[train_size + gap * 2 + val_size:train_size + gap * 2 + val_size + roll_step + data_slice]

        train_data = data.loc[train_dates]
        val_data = data.loc[val_dates]
        test_data = data.loc[test_dates]

        train_dataset = LSTMDataset(train_data)
        val_dataset = LSTMDataset(val_data)
        test_dataset = LSTMDataset(test_data)
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        # for batch_inputs, batch_cov_matrices, batch_labels in val_dataloader:
        #     # print(batch_inputs[0]) 
        #     # print(batch_cov_matrices[0])
        #     # print(batch_labels[0])
        #     print(batch_inputs.shape)       
        #     print(batch_cov_matrices.shape) 
        #     print(batch_labels.shape)
        #     break    

        # model = FactorModel(input_dim=44, hidden_dim=10, output_dim=4, lower=0.05, upper=0.35)
        model = LSTMModel(input_dim=4, lstm_hidden_dim=44, fc_hidden_dim=10, output_dim=4, lower=0.05, upper=0.25)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses, val_losses, best_model = train_model(train_dataloader, val_dataloader, model, optimizer, epochs=epochs, early_stopping=early_stopping)
        torch.save(best_model, f'./test/{test_dates[-roll_step:][0].strftime("%Y%m%d")}-{test_dates[-1].strftime("%Y%m%d")},({round(train_losses[val_losses.index(min(val_losses))],2)},{round(min(val_losses),2)}).pth')

        plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'learning_curve_{test_dates[-roll_step:][0].strftime("%Y%m%d")}-{test_dates[-1].strftime("%Y%m%d")}.png')
        plt.close()

        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            for batch_features, batch_Q_sqrt, _ in test_dataloader:
                weights = model(batch_features, batch_Q_sqrt)
                weights = pd.DataFrame(weights.cpu().numpy(), index=test_data.index.get_level_values(0).drop_duplicates()[-roll_step:], columns=test_data.index.get_level_values(1).drop_duplicates())
                predictions = pd.concat([predictions, weights], axis=0)

        i += roll_step
        
    predictions.to_csv('./test/Weights.xlsx')
    