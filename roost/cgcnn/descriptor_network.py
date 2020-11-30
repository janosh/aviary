from torch import nn

from roost.segments import MeanPooling

from .conv_layer import ConvLayer


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    CrystalGraphConvNet Model.
    """

    def __init__(self, elem_emb_len, nbr_fea_len, elem_fea_len=64, n_graph=4):
        super().__init__()

        self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        self.convs = nn.ModuleList(
            [ConvLayer(elem_fea_len, nbr_fea_len) for _ in range(n_graph)]
        )

        self.pooling = MeanPooling()

    def forward(self, atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
            Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # NOTE required to match the reference implementation
        crys_fea = nn.functional.softplus(crys_fea)

        return crys_fea
