import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from preprocess import tensor_minmax

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