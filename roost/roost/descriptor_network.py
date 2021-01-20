import torch
from torch import nn

from roost.segments import AttentionPooling, SimpleNet

from .message_layer import MessageLayer


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
    ):
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        graph_layers = [
            MessageLayer(elem_fea_len, elem_heads, elem_gate, elem_msg)
            for _ in range(n_graph)
        ]
        self.graphs = nn.ModuleList(graph_layers)

        # define a global pooling function for materials
        cry_pool_layers = [
            AttentionPooling(
                gate_nn=SimpleNet([elem_fea_len, *cry_gate, 1]),
                message_nn=SimpleNet([elem_fea_len, *cry_msg, elem_fea_len]),
            )
            for _ in range(cry_heads)
        ]
        self.cry_pool = nn.ModuleList(cry_pool_layers)

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        return torch.stack(head_fea).mean(dim=0)
