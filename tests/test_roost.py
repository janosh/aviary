import pytest

from aviary.roost import CompositionData, Roost, collate_batch
from aviary.utils import (
    classification_test,
    make_model_dir,
    regression_test,
    train_ensemble,
    train_single,
)

from . import get_params


@pytest.mark.timeout(15)
def test_roostclassification_robust():

    task = "classification"
    data_file = "bandgap-binary.csv"
    emb_file = "matscholar-embedding.json"

    train_kwargs, data_params, test_set = get_params(
        Roost, task, CompositionData, collate_batch, data_file, emb_file
    )

    model_name = "tests/test_roostclassification_robust"
    model_dir = make_model_dir(model_name)
    train_kwargs["model_dir"] = model_dir

    train_single(**train_kwargs)

    # test picking up training with training_ensemble()
    # ensures both the function and resuming training work
    train_ensemble(ensemble_folds=1, **train_kwargs)

    acc, roc_auc, precision, recall, fscore = classification_test(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=1,
        test_set=test_set,
        data_params=data_params,
        robust=True,
    )

    # standard values after 1 epoch
    # - Accuracy  : 0.3871
    # - ROC-AUC   : 0.8687
    # - Precision : 0.1498
    # - Recall    : 0.3871
    # - F-score   : 0.2161
    assert acc > 0.38
    assert roc_auc > 0.86
    assert precision > 0.14
    assert recall > 0.38
    assert fscore > 0.21


@pytest.mark.timeout(10)
def test_roostregression_robust():

    task = "regression"
    data_file = "expt-non-metals.csv"
    emb_file = "matscholar-embedding.json"

    train_kwargs, data_params, test_set = get_params(
        Roost, task, CompositionData, collate_batch, data_file, emb_file
    )
    model_name = "tests/test_roostregression_robust"
    model_dir = make_model_dir(model_name)
    train_kwargs["model_dir"] = model_dir

    train_single(**train_kwargs)

    # test picking up training with training_ensemble()
    # ensures both the function and resuming training work
    train_ensemble(ensemble_folds=1, **train_kwargs)

    r2, mae, rmse = regression_test(
        model_class=Roost,
        model_dir=model_dir,
        ensemble_folds=1,
        test_set=test_set,
        data_params=data_params,
        robust=True,
    )

    # standard values after 1 epoch
    # - R2 Score: -0.0106
    # - MAE: 1.0700
    # - RMSE: 1.5290
    assert r2 > -0.1
    assert mae < 1.15
    assert rmse < 1.6


@pytest.mark.timeout(12)
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
    model_name = "tests/test_single_roostregression_robust"
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
