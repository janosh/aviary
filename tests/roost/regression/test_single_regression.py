from shutil import rmtree

from roost.roost import Roost
from roost.utils import make_model_dir, results_regression, train_ensemble
from tests import data_params_test, data_params_train, setup_params
from tests.roost.regression import model_params, test_set, train_set


def test_single_roost_regression_robust():

    model_name = "test_single_roost_regression_robust"
    model_dir = make_model_dir(model_name)

    train_ensemble(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=1,
        epochs=2,
        train_set=train_set,
        val_set=test_set,
        log=False,
        data_params=data_params_train,
        setup_params=setup_params,
        model_params=model_params,
    )

    r2, mae, rmse = results_regression(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=1,
        test_set=test_set,
        data_params=data_params_test,
        robust=True,
        device=setup_params["device"],
        eval_type="checkpoint",
    )

    # standard values after 2 epochs
    # - R2 Score: 0.0137
    # - MAE: 1.0645
    # - RMSE: 1.5087
    assert r2 > -0.01
    assert mae < 1.1
    assert rmse < 1.6

    rmtree(model_dir)
