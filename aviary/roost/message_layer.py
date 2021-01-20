import torch
from torch import nn

from aviary.segments import AttentionPooling, SimpleNet


class MessageLayer(nn.Module):
    """
    Message Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        super().__init__()

        # Pooling and Output
        pool_layers = [
            AttentionPooling(
                gate_nn=SimpleNet([2 * elem_fea_len, *elem_gate, 1]),
                message_nn=SimpleNet([2 * elem_fea_len, *elem_msg, elem_fea_len]),
            )
            for _ in range(elem_heads)
        ]
        self.pooling = nn.ModuleList(pool_layers)

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elements in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbors to get elements
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights))

        # average the attention heads
        fea = torch.stack(head_fea).mean(dim=0)

        return fea + elem_in_fea
