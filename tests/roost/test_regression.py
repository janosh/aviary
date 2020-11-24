from shutil import rmtree

from roost.roost import Roost
from roost.utils import (
    make_model_dir,
    results_regression,
    train_ensemble,
    train_single,
)
from tests import data_params_test
from tests.roost import get_params_data


def test_roost_regression_robust():

    train_kwargs, test_set = get_params_data("regression", "expt-non-metals.csv")

    model_name = "test_roost_regression_robust"
    model_dir = make_model_dir(model_name)
    train_kwargs["model_dir"] = model_dir

    train_single(**train_kwargs)

    # test picking up training with training_ensemble()
    # ensures both the function and resuming training work
    train_ensemble(ensemble_folds=1, **train_kwargs)

    r2, mae, rmse = results_regression(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=1,
        test_set=test_set,
        data_params=data_params_test,
        robust=True,
    )

    # standard values after 1 epoch
    # - R2 Score: -0.0106
    # - MAE: 1.0700
    # - RMSE: 1.5290
    assert r2 > -0.1
    assert mae < 1.1
    assert rmse < 1.6

    rmtree(model_dir)
