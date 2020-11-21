import os
from functools import wraps
from typing import Callable

from roost.core import ROOT

from .init_model import init_model
from .results_classification import results_classification
from .results_regression import results_regression
from .train_model import train_ensemble, train_single


def make_model_dir(model_name, ensemble=1, run_id="run_1"):
    """Generate the correct directories for saving model checkpoints,
    TensorBoard runs and other output depending on whether using ensembles
    or single models.
    """
    model_dir = f"{ROOT}/models/{model_name}"

    if ensemble > 1:
        for idx in range(ensemble):
            os.makedirs(f"{model_dir}/ens_{idx}", exist_ok=True)
    else:
        # bake run_id into the return value if in single-model mode
        model_dir += f"/{run_id}"
        os.makedirs(model_dir, exist_ok=True)

    return model_dir


def interruptable(func: Callable):
    """Allows gracefully aborting calls to the decorated function with ctrl + c."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print(f"\nDetected KeyboardInterrupt: Aborting {func.__name__}()")

    return wrapped
