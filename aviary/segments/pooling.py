import torch
from torch import nn
from torch_scatter import scatter_add, scatter_max


class AttentionPooling(nn.Module):
    """Softmax attention layer"""

    def __init__(self, gate_nn, message_nn):
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, index, weights=None):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        if weights is not None:
            gate = (weights ** self.pow) * gate
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out
