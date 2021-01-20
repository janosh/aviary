from aviary.base import BaseModel
from aviary.segments import ResidualNet

from . import DescriptorNetwork


class Roost(BaseModel):
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
        out_hidden=[1024, 512, 256, 128, 64],
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

        model_params = {
            "task": task,
            "robust": robust,
            "n_targets": n_targets,
            "out_hidden": out_hidden,
        }
        self.model_params.update({**model_params, **desc_dict})

        # define an output neural network
        output_dim = 2 * n_targets if self.robust else n_targets

        self.output_nn = ResidualNet(dims=[elem_fea_len, *out_hidden, output_dim])

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):
        """ Forward pass through the material_nn and output_nn """

        crys_fea = self.material_nn(
            elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
        )

        # apply neural network to map from learned features to target
        return self.output_nn(crys_fea)
