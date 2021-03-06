import pytest

from aviary.cgcnn import CrystalGraphConvNet, CrystalGraphData, collate_batch
from aviary.utils import (
    classification_test,
    make_model_dir,
    regression_test,
    train_ensemble,
    train_single,
)

from . import get_params


@pytest.mark.timeout(5)
def test_cgcnn_classification_robust():

    task = "classification"
    data_file = "cgcnn-clf-is-metal.csv"
    emb_file = "cgcnn-embedding.json"

    train_kwargs, data_params, test_set = get_params(
        CrystalGraphConvNet, task, CrystalGraphData, collate_batch, data_file, emb_file
    )

    model_name = "tests/test_cgcnn_classification_robust"
    model_dir = make_model_dir(model_name)
    train_kwargs["model_dir"] = model_dir

    train_single(**train_kwargs)

    # test picking up training with training_ensemble()
    # ensures both the function and resuming training work
    train_ensemble(ensemble_folds=1, **train_kwargs)

    acc, roc_auc, precision, recall, fscore = classification_test(
        model_class=CrystalGraphConvNet,
        model_dir=model_dir,
        ensemble_folds=1,
        test_set=test_set,
        data_params=data_params,
        robust=True,
    )

    # standard values after 1 epoch
    # - Accuracy  : 0.6000
    # - ROC-AUC   : 0.4583
    # - Precision : 0.3600
    # - Recall    : 0.6000
    # - F-score   : 0.4500
    assert acc > 0.55
    assert roc_auc > 0.45
    assert precision > 0.35
    assert recall > 0.55
    assert fscore > 0.4


@pytest.mark.timeout(5)
def test_cgcnn_regression_robust():

    task = "regression"
    data_file = "cgcnn-regr-e-formation.csv"
    emb_file = "cgcnn-embedding.json"

    train_kwargs, data_params, test_set = get_params(
        CrystalGraphConvNet, task, CrystalGraphData, collate_batch, data_file, emb_file
    )
    model_name = "tests/test_cgcnn_regression_robust"
    model_dir = make_model_dir(model_name)
    train_kwargs["model_dir"] = model_dir

    train_single(**train_kwargs)

    # test picking up training with training_ensemble()
    # ensures both the function and resuming training work
    train_ensemble(ensemble_folds=1, **train_kwargs)

    r2, mae, rmse = regression_test(
        model_class=CrystalGraphConvNet,
        model_dir=model_dir,
        ensemble_folds=1,
        test_set=test_set,
        data_params=data_params,
        robust=True,
    )

    # standard values after 1 epoch
    # - R2 Score: -4.8471
    # - MAE: 3.2717
    # - RMSE: 3.5027
    assert r2 > -4.9
    assert mae < 3.3
    assert rmse < 3.6
