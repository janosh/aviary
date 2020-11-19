import json
from os.path import abspath, dirname

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from torch.nn.functional import softmax

ROOT = dirname(dirname(abspath(__file__)))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RegressionMetrics:
    """Computes and stores average metrics for regression tasks"""

    def __init__(self):
        self.rmse_meter = AverageMeter()
        self.mae_meter = AverageMeter()

    def update(self, pred, target):
        mae_error = mae(pred, target)
        self.mae_meter.update(mae_error)

        rmse_error = np.sqrt(mse(pred, target))
        self.rmse_meter.update(rmse_error)

    @property
    def metric_dict(self):
        return {"MAE": self.mae_meter.avg, "RMSE": self.rmse_meter.avg}


class ClassificationMetrics:
    """Computes and stores average metrics for classification tasks"""

    def __init__(self):
        self.acc_meter = AverageMeter()
        self.fscore_meter = AverageMeter()

    def update(self, pred, target):
        acc = accuracy_score(target, np.argmax(pred, axis=1))
        self.acc_meter.update(acc)

        fscore = f1_score(target, np.argmax(pred, axis=1), average="weighted")
        self.fscore_meter.update(fscore)

    @property
    def metric_dict(self):
        return {"Acc": self.acc_meter.avg, "F1": self.fscore_meter.avg}


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = tensor.mean(dim, keepdim)
        self.std = tensor.std(dim, keepdim)

    def norm(self, tensor):
        assert [self.mean, self.std] != [None, None], "Normalizer must be fit first"
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        assert [self.mean, self.std] != [None, None], "Normalizer must be fit first"
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()


class Featurizer:
    """
    Base class for featurising nodes and edges.
    """

    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(self._embedding[list(self._embedding.keys())[0]])


class LoadFeaturizer(Featurizer):
    """
    Initialize a featurizer from a JSON file.

    Parameters
    ----------
    embedding_file: str
        The path to the .json file
    """

    def __init__(self, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = set(embedding.keys())
        super().__init__(allowed_types)
        for key, value in embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def save_checkpoint(state, is_best, model_name, run_id):
    """
    Saves a checkpoint and overwrites the best model when is_best==True
    """
    checkpoint = f"{ROOT}/models/{model_name}/checkpoint-r{run_id}.pth.tar"
    torch.save(state, checkpoint)

    if is_best:
        best = f"{ROOT}/models/{model_name}/best-r{run_id}.pth.tar"
        torch.save(state, best)


def RobustL1Loss(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = 2 ** 0.5 * torch.abs(output - target) * torch.exp(-log_std) + log_std
    return loss.mean()


def RobustL2Loss(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = 0.5 * torch.pow(output - target, 2.0) * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)


def sampled_softmax(pre_logits, log_std, samples=10):
    """
    Draw samples from gaussian distributed pre-logits and use these to estimate
    a mean and aleatoric uncertainty.
    """
    # NOTE here as we do not risk dividing by zero should we really be
    # predicting log_std or is there another way to deal with negative numbers?
    # This choice may have an unknown effect on the calibration of the uncertainties
    sam_std = torch.exp(log_std).repeat_interleave(samples, dim=0)
    # TODO here we are normally distributing the samples even if the loss
    # uses a different prior?
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + torch.mul(
        epsilon, sam_std
    )
    logits = softmax(pre_logits, dim=1).view(len(log_std), samples, -1)
    return torch.mean(logits, dim=1)
