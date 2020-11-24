from shutil import rmtree

from roost.roost import Roost
from roost.utils import make_model_dir, results_regression, train_single
from tests import data_params_test
from tests.roost import get_params_data


def test_train_individually_evaluate_ensemble():
    """Test training multiple models individually but evaluating them as ensemble.
    Simulates faster method of training models in parallel on HPC architectures.
    """
    train_kwargs, test_set = get_params_data("regression", "expt-non-metals.csv")
    model_name = "test_single_roost_regression_robust"
    model_dir = make_model_dir(model_name, ensemble=2)

    for run_id in ["ens_0", "ens_1"]:
        train_single(model_dir=model_dir + f"/{run_id}", **train_kwargs)

    r2, mae, rmse = results_regression(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=2,
        test_set=test_set,
        data_params=data_params_test,
        robust=True,
    )

    # standard values after 1 epoch
    # - R2 Score: -0.01057
    # - MAE: 1.0776
    # - RMSE: 1.5200
    assert r2 > -0.02
    assert mae < 1.1
    assert rmse < 1.6

    rmtree(model_dir)
