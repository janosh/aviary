from roost.roost import CompositionData, Roost, collate_batch
from roost.utils import (
    make_model_dir,
    classification_test,
    train_ensemble,
    train_single,
)
from tests import get_params


def test_roost_classification_robust():

    task = "classification"
    data_file = "bandgap-binary.csv"
    emb_file = "matscholar-embedding.json"

    train_kwargs, data_params, test_set = get_params(
        Roost, task, CompositionData, collate_batch, data_file, emb_file
    )

    model_name = "tests/test_roost_classification_robust"
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
