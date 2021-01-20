import ast
import functools
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from roost.core import LoadFeaturizer


class WyckoffData(Dataset):
    """
    The WrenData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(self, data_path, sym_path, fea_path, task):

        assert os.path.exists(data_path), f"{data_path} does not exist!"
        # make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

        assert os.path.exists(fea_path), f"{fea_path} does not exist!"
        self.atom_features = LoadFeaturizer(fea_path)
        assert os.path.exists(sym_path), f"{sym_path} does not exist!"
        self.sym_features = LoadFeaturizer(sym_path)

        # TODO clean this up to use package reasources
        with open("data/wren/relab.json") as f:
            self.relab_dict = json.load(f)

        for key, val in self.relab_dict.items():
            self.relab_dict[key] = [
                {int(sk): sv for sk, sv in lst.items()} for lst in val
            ]

        self.elem_emb_len = self.atom_features.embedding_size
        self.sym_fea_dim = self.sym_features.embedding_size
        self.task = task
        self.n_targets = np.max(self.df[self.df.columns[2]].values) + 1

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        Returns
        -------
        atom_weights: torch.Tensor shape (M, 1)
            weights of atoms in the material
        atom_fea: torch.Tensor shape (M, n_fea)
            features of atoms in the material
        self_fea_idx: torch.Tensor shape (M*M, 1)
            list of self indices
        nbr_fea_idx: torch.Tensor shape (M*M, 1)
            list of neighbor indices
        target: torch.Tensor shape (1,)
            target value for material
        cry_id: torch.Tensor shape (1,)
            input id for the material
        """
        # cry_id, composition, target = self.id_prop_data[idx]
        cry_id, composition, target, swyks = self.df.iloc[idx]
        weights, elements, aug_wyks = parse_wren(swyks, self.relab_dict)

        weights = np.atleast_2d(weights).T / np.sum(weights)
        assert (
            len(elements) != 1
        ), f"crystal {cry_id}: {composition}, {swyks} is a pure system"
        try:
            atom_fea = np.vstack([self.atom_features.get_fea(el) for el in elements])
            sym_fea = np.vstack(
                [self.sym_features.get_fea(wyk) for wyks in aug_wyks for wyk in wyks]
            )
        except AssertionError:
            print(f"failed to process {cry_id}: {composition}")
            raise

        n_wyks = len(elements)
        env_idx = list(range(n_wyks))
        self_fea_idx = []
        nbr_fea_idx = []
        for i in range(n_wyks):
            self_fea_idx += [i] * (n_wyks - 1)
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

        self_aug_fea_idx = []
        nbr_aug_fea_idx = []
        n_aug = len(aug_wyks)
        for i in range(n_aug):
            self_aug_fea_idx += [x + i * n_wyks for x in self_fea_idx]
            nbr_aug_fea_idx += [x + i * n_wyks for x in nbr_fea_idx]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        sym_fea = torch.Tensor(sym_fea)
        self_fea_idx = torch.LongTensor(self_aug_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_aug_fea_idx)
        if self.task == "regression":
            target = torch.Tensor([target])
        elif self.task == "classification":
            target = torch.LongTensor([target])

        return (
            (atom_weights, atom_fea, sym_fea, self_fea_idx, nbr_fea_idx),
            target,
            composition,
            cry_id,
        )


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom"s M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_sym_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    aug_cry_idx = []
    batch_target = []
    batch_comp = []
    batch_cry_ids = []

    aug_count = 0
    cry_base_idx = 0
    for (
        i,
        (
            (atom_weights, atom_fea, sym_fea, self_fea_idx, nbr_fea_idx),
            target,
            comp,
            cry_id,
        ),
    ) in enumerate(dataset_list):
        # number of atoms for this crystal
        n_el = atom_fea.shape[0]
        n_i = sym_fea.shape[0]
        n_aug = int(float(n_i) / float(n_el))

        # batch the features together
        batch_atom_weights.append(atom_weights.repeat((n_aug, 1)))
        batch_atom_fea.append(atom_fea.repeat((n_aug, 1)))
        batch_sym_fea.append(sym_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        # print(torch.tensor(range(i, i+n_aug)).size())
        crystal_atom_idx.append(
            torch.tensor(list(range(aug_count, aug_count + n_aug))).repeat_interleave(
                n_el
            )
        )
        aug_cry_idx.append(torch.tensor([i] * n_aug))

        # batch the targets and ids
        batch_target.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

        # increment the id counter
        aug_count += n_aug
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_sym_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
            torch.cat(aug_cry_idx),
        ),
        torch.stack(batch_target, dim=0),
        batch_comp,
        batch_cry_ids,
    )


def parse_wren(swyk_list, relab_dict):
    """"""
    swyk_list = ast.literal_eval(swyk_list)

    mult_list = []
    ele_list = []
    wyk_list = []

    for swyk in swyk_list:
        # mult, ele, wyk = swyk.split("_")
        ele, wyk = swyk.split(" @ ")
        mult, _, spg = wyk.split("-")
        mult_list.append(float(mult))
        ele_list.append(ele)
        wyk_list.append(wyk)

    aug_wyks = []
    for trans in relab_dict[spg]:
        aug_wyks.append(
            tuple(",".join(wyk_list).translate(str.maketrans(trans)).split(","))
        )

    aug_wyks = list(set(aug_wyks))

    return mult_list, ele_list, aug_wyks
