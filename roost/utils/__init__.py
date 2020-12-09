from functools import wraps
from typing import Callable

import torch

from .classification_test import classification_test
from .init_model import init_model
from .io import ROOT, bold, make_model_dir
from .mat_proj import API_KEY, fetch_mp
from .regression_test import regression_test
from .train_model import train_ensemble, train_single


def run_test(task, *args, **kwargs):
    assert task in ["regression", "classification"], f"invalid task: {task}"
    func = regression_test if task == "regression" else classification_test
    return func(*args, **kwargs)


def interruptable(func: Callable):
    """Allows gracefully aborting calls to the decorated function with ctrl + c."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print(f"\nDetected KeyboardInterrupt: Aborting {func.__name__}()")

    return wrapped


def pearsonr(x, y):
    """Taken from https://github.com/pytorch/pytorch/issues/1254.
    Mimics `scipy.stats.pearsonr`. x and y assumed to be 1d tensors.
    Returns the Pearson correlation coefficient (float tensor) between x and y.

    Scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code: https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033

    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> scipy_corr = scipy.stats.pearsonr(x, y)[0]
        >>> torch_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(scipy_corr, torch_corr)
    """
    xm = x - x.mean()
    ym = y - y.mean()
    r_num = xm @ ym
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val
