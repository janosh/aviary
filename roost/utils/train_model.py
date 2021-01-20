from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from roost.utils import init_model


def train_single(
    model_class,
    model_dir,
    epochs,
    train_set,
    val_set,
    log,
    verbose,
    data_params,
    setup_params,
    model_params,
    swa=None,
):
    """Train a single model"""

    train_generator = DataLoader(train_set, **data_params)

    if val_set is not None:
        data_params.update({"batch_size": 16 * data_params["batch_size"]})
        val_generator = DataLoader(val_set, **data_params)
    else:
        val_generator = None

    model, criterion, optimizer, scheduler, normalizer = init_model(
        model_class, model_dir, model_params, swa=swa, **setup_params
    )

    if model.task == "regression":
        sample_target = torch.Tensor(
            train_set.dataset.df.iloc[train_set.indices, 2].values
        )
        if normalizer.mean is None:  # normalizer hasn't been fit yet
            normalizer.fit(sample_target)
        print(f"Dummy MAE: {(sample_target - normalizer.mean).abs().mean():.4f}")

    if log:
        now = f"{datetime.now():%d-%m-%Y_%H-%M-%S}"
        writer = SummaryWriter(f"{model_dir}/runs/{now}")
    else:
        writer = None

    if (val_set is not None) and (model.best_val_score is None):
        # get validation baseline
        with torch.no_grad():
            val_metrics = model.evaluate(
                val_generator,
                criterion,
                optimizer=None,
                normalizer=normalizer,
                action="val",
                verbose=verbose,
            )
        val_score = val_metrics[model.val_score_name.lower()]
        print(f"Validation Baseline: {model.val_score_name} = {val_score:.3f}\n")
        model.best_val_score = val_score

    model.fit(
        train_generator,
        val_generator,
        optimizer,
        scheduler,
        epochs,
        criterion,
        normalizer,
        model_dir,
        writer=writer,
        verbose=verbose,
    )


def train_ensemble(ensemble_folds, model_dir, *args, **kwargs):
    """Train multiple models in sequence

    Note: It's faster to train multiple models at once in an array job and
    specify run_ids like ens_0, ens_1, ... than to train in sequence using
    this function. Either way the ensemble can be evaluated using
    regression_test/_classification afterwards.
    """

    # allow using this function even in single-model mode
    if ensemble_folds == 1:
        train_single(*args, model_dir=model_dir, **kwargs)
        return

    for idx in range(ensemble_folds):
        train_single(*args, model_dir=model_dir + f"/ens_{idx}", **kwargs)
