from typing import List

import torch
from torch import nn
from torch_mnf.layers import MNFLinear
from torch_scatter import scatter_add, scatter_max, scatter_mean


class MeanPooling(nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, index):

        mean = scatter_mean(x, index, dim=0)

        return mean


class SumPooling(nn.Module):
    """
    sum pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, index):

        mean = scatter_add(x, index, dim=0)

        return mean


class AttentionPooling(nn.Module):
    """
    softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x, index):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn((1)))

    def forward(self, x, index, weights):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self, dims: List[int], activation=nn.LeakyReLU, batchnorm: bool = False
    ):
        super().__init__()
        output_dim = dims.pop()

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(
        self, dims: List[int], activation=nn.ReLU, batchnorm: bool = True, use_mnf=False
    ):
        super().__init__()

        output_dim = dims.pop()

        fc = MNFLinear if use_mnf else nn.Linear

        self.fcs = nn.ModuleList(
            [fc(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        bn_layers = [
            nn.BatchNorm1d(dims[i + 1]) if batchnorm else nn.Identity()
            for i in range(len(dims) - 1)
        ]
        self.bns = nn.ModuleList(bn_layers)

        self.res_fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)]
        )

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        return self.fc_out(x)
