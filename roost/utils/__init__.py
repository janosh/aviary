from functools import wraps
from typing import Callable

from .init_model import init_model
from .io import ROOT, bold, make_model_dir
from .mat_proj import API_KEY, fetch_mp
from .results_classification import results_classification
from .results_regression import results_regression
from .train_model import train_ensemble, train_single


def interruptable(func: Callable):
    """Allows gracefully aborting calls to the decorated function with ctrl + c."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print(f"\nDetected KeyboardInterrupt: Aborting {func.__name__}()")

    return wrapped
