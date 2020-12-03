from roost.cgcnn import CrystalGraphConvNet, CrystalGraphData, collate_batch
from roost.utils import (
    make_model_dir,
    regression_test,
    train_ensemble,
    train_single,
)
from tests import get_params


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
