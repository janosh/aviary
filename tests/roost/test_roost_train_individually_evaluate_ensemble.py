from roost.roost import CompositionData, Roost, collate_batch
from roost.utils import make_model_dir, regression_test, train_single
from tests import get_params


def test_train_individually_evaluate_ensemble():
    """Test training multiple models individually but evaluating them as ensemble.
    Simulates faster method of training models in parallel on HPC architectures.
    """

    task = "regression"
    data_file = "expt-non-metals.csv"
    emb_file = "matscholar-embedding.json"

    train_kwargs, data_params, test_set = get_params(
        Roost, task, CompositionData, collate_batch, data_file, emb_file
    )
    model_name = "tests/test_single_roost_regression_robust"
    model_dir = make_model_dir(model_name, ensemble=2)

    for run_id in ["ens_0", "ens_1"]:
        train_single(model_dir=model_dir + f"/{run_id}", **train_kwargs)

    r2, mae, rmse = regression_test(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=2,
        test_set=test_set,
        data_params=data_params,
        robust=True,
    )

    # standard values after 1 epoch
    # - R2 Score: -0.01057
    # - MAE: 1.0776
    # - RMSE: 1.5200
    assert r2 > -0.03
    assert mae < 1.15
    assert rmse < 1.6
