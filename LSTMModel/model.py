import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from preprocess import tensor_minmax

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