from shutil import rmtree

from roost.roost import Roost
from roost.utils import make_model_dir, results_classification, train_ensemble
from tests import data_params_test, data_params_train, setup_params
from tests.roost.classification import model_params, test_set, train_set


def test_single_roost_classification_robust():

    model_name = "test_single_roost_classification_robust"
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

    acc, roc_auc, precision, recall, fscore = results_classification(
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
    # - Accuracy : 0.8395
    # - ROC-AUC: 0.9595
    # - Precision: 0.8702
    # - Recall: 0.8395
    # - F-score: 0.8415
    assert acc > 0.83
    assert roc_auc > 0.95
    assert precision > 0.86
    assert recall > 0.83
    assert fscore > 0.84

    rmtree(model_dir)
