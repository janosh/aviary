from torch import nn

from aviary.base import BaseModel
from aviary.segments import ResidualNet

from .descriptor_network import DescriptorNetwork


class Wren(BaseModel):
    """
    Like Roost, the Wren model is comprised of message-passing graph
    layers that feed into a fully-connected network.

    The message-passing layers can be seen as featurization layers.
    The graphs are used to represent the stoichiometry of inorganic
    materials in a trainable manner. This makes them systematically
    improvable with more data.
    """

    def __init__(
        self,
        task,
        robust,
        n_targets,
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
        out_hidden=[1024, 512, 256, 128, 64],
        **kwargs
    ):
        super().__init__(task=task, robust=robust, n_targets=n_targets, **kwargs)

        # TODO find a more elegant way to structure this unpacking then
        # a dictionary seems like a non-optimal solution.
        self.model_params.update()

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "sym_emb_len": sym_emb_len,
            "sym_fea_len": sym_fea_len,
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
                **desc_dict,
            }
        )

        # define an output neural network
        output_dim = 2 * n_targets if self.robust else n_targets

        self.output_nn = ResidualNet([elem_fea_len + sym_fea_len] + out_hidden)
        # NOTE separating out the final rect_linear unit allows easy access to
        # the final embedding space
        self._linear = nn.Linear(out_hidden[-1], output_dim)

    def forward(
        self,
        elem_weights,
        elem_fea,
        sym_fea,
        self_fea_idx,
        nbr_fea_idx,
        cry_elem_idx,
        aug_cry_idx,
        **kwargs
    ):
        """
        Forward pass through the material_nn and output_nn
        """
        crys_fea = self.material_nn(
            elem_weights,
            elem_fea,
            sym_fea,
            self_fea_idx,
            nbr_fea_idx,
            cry_elem_idx,
            aug_cry_idx,
        )

        # apply neural network to map from learned features to target
        return self._linear(nn.functional.relu(self.output_nn(crys_fea)))
