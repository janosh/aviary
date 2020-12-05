from functools import wraps
from typing import Callable

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
