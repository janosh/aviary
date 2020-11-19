from functools import lru_cache
from os.path import exists

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from roost.core import LoadFeaturizer

from .utils import parse_roost


class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(self, data_path, fea_path, task):
        assert exists(data_path), f"{data_path} does not exist!"
        # NOTE make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

        assert exists(fea_path), f"{fea_path} does not exist!"
        self.elem_features = LoadFeaturizer(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size
        self.task = task
        if self.task == "regression":
            if self.df.shape[1] - 2 != 1:
                raise NotImplementedError(
                    "Multi-target regression currently not supported"
                )
            self.n_targets = 1
        elif self.task == "classification":
            if self.df.shape[1] - 2 != 1:
                raise NotImplementedError(
                    "One-Hot input not supported please use categorical integer"
                    " inputs for classification i.e. Dog = 0, Cat = 1, Mouse = 2"
                )
            self.n_targets = np.max(self.df[self.df.columns[2]].values) + 1

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=None)  # Cache loaded structures
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
        cry_id, composition, target = self.df.iloc[idx]
        elements, weights = parse_roost(composition)
        weights = np.atleast_2d(weights).T / np.sum(weights)
        assert len(elements) != 1, f"cry-id {cry_id} [{composition}] is a pure system"
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )

        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elements) - 1
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        if self.task == "regression":
            targets = torch.Tensor([float(target)])
        elif self.task == "classification":
            targets = torch.LongTensor([target])

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            targets,
            composition,
            cry_id,
        )
