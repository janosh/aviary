from shutil import rmtree

from roost.roost import Roost
from roost.utils import make_model_dir, results_regression, train_single
from tests import data_params_test, data_params_train, setup_params
from tests.roost.regression import model_params, test_set, train_set


def test_train_individually_evaluate_ensemble():
    """Test training multiple models individually but evaluating them as ensemble.
    Simulates faster method of training models in parallel on HPC architectures.
    """

    model_name = "test_single_roost_regression_robust"
    model_dir = make_model_dir(model_name, ensemble=2)

    for run_id in ["ens_0", "ens_1"]:
        train_single(
            model_class=Roost,
            model_dir=model_dir + f"/{run_id}",
            epochs=2,
            train_set=train_set,
            val_set=test_set,
            log=False,
            verbose=False,
            data_params=data_params_train,
            setup_params=setup_params,
            model_params=model_params,
        )

    r2, mae, rmse = results_regression(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=2,
        test_set=test_set,
        data_params=data_params_test,
        robust=True,
    )

    # standard values after 2 epochs
    # - R2 Score: -0.01057
    # - MAE: 1.0776
    # - RMSE: 1.5200
    assert r2 > -0.02
    assert mae < 1.1
    assert rmse < 1.6

    rmtree(model_dir)
