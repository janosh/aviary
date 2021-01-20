from typing import List

from torch import nn
from torch_mnf.layers import MNFLinear


class ResidualNet(nn.Module):
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
        )  # no bias needed since fcs already has one, keeps param count down

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        return self.fc_out(x)
