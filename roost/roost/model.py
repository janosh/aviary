import torch
from torch import nn

from roost.core import BaseModelClass
from roost.segments import (
    ResidualNetwork,
    SimpleNetwork,
    WeightedAttentionPooling,
)


class Roost(BaseModelClass):
    """
    The Roost model is comprised of a fully connected network
    and message passing graph layers.

    The message passing layers are used to determine a descriptor set
    for the fully connected network. The graphs are used to represent
    the stoichiometry of inorganic materials in a trainable manner.
    This makes them systematically improvable with more data.
    """

    def __init__(
        self,
        task,
        robust,
        n_targets,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        out_hidden=[512, 256, 128, 64, 32],
        use_mnf=False,
        **kwargs
    ):
        super().__init__(task=task, robust=robust, n_targets=n_targets, **kwargs)

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
        }

        self.material_nn = DescriptorNetwork(**desc_dict)

        self.model_params.update(
            {
                "task": task,
                "robust": robust,
                "n_targets": n_targets,
                "out_hidden": out_hidden,
                "use_mnf": use_mnf,
                **desc_dict,
            }
        )

        # define an output neural network
        output_dim = 2 * n_targets if self.robust else n_targets

        # self.output_nn = nn.Linear(elem_fea_len, output_dim)
        self.output_nn = ResidualNetwork(
            dims=[elem_fea_len, *out_hidden, output_dim], use_mnf=use_mnf
        )
        # self.output_nn = SimpleNetwork([elem_fea_len, *out_hidden, output_dim], nn.ReLU)

    def forward(
        self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, repeat=1
    ):
        """
        Forward pass through the material_nn and output_nn
        """
        crys_fea = self.material_nn(
            elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
        )

        # apply neural network to map from learned features to target
        return torch.stack([self.output_nn(crys_fea) for _ in range(repeat)], dim=-1)


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
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
        # self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        # create a list of Message passing layers
        graph_layers = [
            MessageLayer(elem_fea_len, elem_heads, elem_gate, elem_msg)
            for _ in range(n_graph)
        ]
        self.graphs = nn.ModuleList(graph_layers)

        # define a global pooling function for materials
        cry_pool_layers = [
            WeightedAttentionPooling(
                gate_nn=SimpleNetwork([elem_fea_len, *cry_gate, 1]),
                message_nn=SimpleNetwork([elem_fea_len, *cry_msg, elem_fea_len]),
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
                # attnhead(elem_fea, index=cry_elem_idx)
            )

        return torch.stack(head_fea).mean(dim=0)


class MessageLayer(nn.Module):
    """
    Message Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        super().__init__()

        # Pooling and Output
        pool_layers = [
            WeightedAttentionPooling(
                gate_nn=SimpleNetwork([2 * elem_fea_len, *elem_gate, 1]),
                message_nn=SimpleNetwork([2 * elem_fea_len, *elem_msg, elem_fea_len]),
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
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
                # attnhead(self.mean_msg(fea), index=self_fea_idx)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea
        # return fea
