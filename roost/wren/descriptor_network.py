import torch
from torch import nn

from roost.segments import MeanPooling, SimpleNet, WeightedAttentionPooling

from .message_layer import MessageLayer


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        sym_emb_len,
        elem_fea_len=32,
        sym_fea_len=32,
        n_graph=3,
        elem_heads=1,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=1,
        cry_gate=[256],
        cry_msg=[256],
    ):
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.elem_embed = nn.Linear(elem_emb_len, elem_fea_len - 1)
        self.sym_embed = nn.Linear(sym_emb_len, sym_fea_len)

        # create a list of Message passing layers
        fea_len = elem_fea_len + sym_fea_len

        # create a list of Message passing layers
        graph_layers = [
            MessageLayer(fea_len, elem_heads, elem_gate, elem_msg)
            for _ in range(n_graph)
        ]
        self.graphs = nn.ModuleList(graph_layers)

        # define a global pooling function for materials
        cry_pool_layers = [
            WeightedAttentionPooling(
                gate_nn=SimpleNet([fea_len, *cry_gate, 1]),
                message_nn=SimpleNet([fea_len, *cry_msg, fea_len]),
            )
            for _ in range(cry_heads)
        ]
        self.cry_pool = nn.ModuleList(cry_pool_layers)

        self.aug_pool = MeanPooling()

    def forward(
        self,
        elem_weights,
        elem_fea,
        sym_fea,
        self_fea_idx,
        nbr_fea_idx,
        cry_elem_idx,
        aug_cry_idx,
    ):
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
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
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

        # embed the original features into the graph layer description
        elem_fea = self.elem_embed(elem_fea)
        sym_fea = self.sym_embed(sym_fea)

        # do this so that we can examine the embeddings without
        # influence of the weights
        elem_fea = torch.cat([elem_fea, sym_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        cry_fea = self.aug_pool(torch.stack(head_fea).mean(dim=0), aug_cry_idx)

        return cry_fea
