import torch
from torch import nn
from torch.nn.functional import softplus
from torch_scatter import scatter_add


class ConvLayer(nn.Module):
    """ Convolutional operation on graphs """

    def __init__(self, elem_fea_len, nbr_fea_len):
        """
        Args:
            elem_fea_len (int): Number of atom hidden features
            nbr_fea_len (int): Number of bond features
        """
        super().__init__()
        self.elem_fea_len = elem_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.elem_fea_len + self.nbr_fea_len, 2 * self.elem_fea_len
        )
        self.bn1 = nn.BatchNorm1d(2 * self.elem_fea_len)
        self.bn2 = nn.BatchNorm1d(self.elem_fea_len)

    def forward(self, atom_in_fea, nbr_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, elem_fea_len)
            Atom hidden features after convolution

        """
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        atom_self_fea = atom_in_fea[self_fea_idx, :]

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=1)

        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)

        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = torch.sigmoid(filter_fea)
        core_fea = softplus(core_fea)

        # take the elementwise product of the filter and core
        nbr_msg = filter_fea * core_fea
        nbr_sumed = scatter_add(nbr_msg, self_fea_idx, dim=0)  # sum pooling

        nbr_sumed = self.bn2(nbr_sumed)
        out = softplus(atom_in_fea + nbr_sumed)

        return out
