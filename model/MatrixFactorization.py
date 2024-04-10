import torch
import torch.nn as nn
import torch.optim as optim
from tensorly.tucker_tensor import tucker_to_tensor


class Tucker(nn.Module):
    def __init__(self, N, K=128):
        super(Tucker, self).__init__()
        self.shape = [N, N, 24 * 7]
        self.ranks = [K, K, K]
        self.core = nn.Parameter(torch.randn(K, K, K))
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(self.shape[i], self.ranks[i])) for i in range(3)])

    def forward(self):
        return tucker_to_tensor([self.core, self.factors])


class MatrixFactorizationPre(nn.Module):
    def __init__(self, N, K=128):
        super(MatrixFactorizationPre, self).__init__()
        self.V = nn.Parameter(torch.randn(N, K))
        self.W = nn.Parameter(torch.randn(K, K))

    def forward(self):
        VWT = torch.matmul(self.V, torch.matmul(self.W, self.V.t()))
        return VWT
