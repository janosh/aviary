import json
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import softmax


class Normalizer:
    """Normalize a tensor and restore it later."""

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, tensor: Tensor, dim: int = 0) -> None:
        self.mean = tensor.mean(dim)
        self.std = tensor.std(dim)
        assert (self.std != 0).all(), "self.std has 0 entries, cannot divide by 0"

    def norm(self, tensor: Tensor) -> Tensor:
        assert self.is_fit, "Normalizer must be fit first"
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: Tensor) -> Tensor:
        assert self.is_fit, "Normalizer must be fit first"
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict) -> None:
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @property
    def is_fit(self) -> bool:
        return [self.mean, self.std] != [None, None]


class Featurizer:
    """Base class for featurizing nodes and edges in a crystal graph."""

    def __init__(self, allowed_types: Iterable[str]) -> None:
        """
        Args:
            allowed_types (Iterable[str]): names of element names for which
            to store embeddings
        """
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key: str) -> np.ndarray:
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict: dict) -> None:
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self) -> dict:
        return self._embedding

    @property
    def embedding_size(self) -> int:
        return len(list(self._embedding.values())[0])

    @classmethod
    def from_json(cls, embedding_file: str) -> "Featurizer":
        with open(embedding_file) as file:
            embedding = json.load(file)
        allowed_types = set(embedding.keys())
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance


def save_checkpoint(state, is_best, model_dir):
    """
    Saves a checkpoint and overwrites the best model when is_best==True.
    """
    torch.save(state, f"{model_dir}/checkpoint.pth.tar")

    if is_best:
        torch.save(state, f"{model_dir}/best.pth.tar")


def RobustL1Loss(output, log_std, target):
    """Robust L1 loss using a Lorentzian prior.
    Allows for aleatoric uncertainty estimation.
    """
    loss = 2 ** 0.5 * (output - target).abs() / log_std.exp() + log_std
    return loss.mean()


def RobustL2Loss(output, log_std, target):
    """Robust L2 loss using a gaussian prior.
    Allows for aleatoric uncertainty estimation.
    """
    loss = 0.5 * (output - target) ** 2 / (2 * log_std).exp() + log_std
    return loss.mean()


def sampled_softmax(pre_logits, log_std, samples=10):
    """Draw samples from Gaussian distributed pre-logits and use these to
    estimate a mean and aleatoric uncertainty.
    """
    # NOTE here as we do not risk dividing by zero should we really be
    # predicting log_std or is there another way to deal with negative numbers?
    # This choice may have an unknown effect on the calibration of the uncertainties
    sam_std = log_std.exp().repeat_interleave(samples, dim=0)
    # TODO here we are normally distributing the samples even if the loss
    # uses a different prior?
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + epsilon * sam_std
    logits = softmax(pre_logits, dim=1).view(len(log_std), samples, -1)
    return logits.mean(dim=1)
