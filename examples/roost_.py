# don't name this file roost.py as it will shadow the roost package/folder itself
import argparse

import torch
from sklearn.model_selection import train_test_split as split

from examples.common_cli_args import add_common_args
from roost.roost import CompositionData, Roost, collate_batch
from roost.utils import (
    make_model_dir,
    results_classification,
    results_regression,
    train_ensemble,
)


def main(
    data_path,
    fea_path,
    task,
    loss,
    robust,
    model_name="roost",
    elem_fea_len=64,
    n_graph=3,
    ensemble=1,
    run_id=1,
    data_seed=42,
    epochs=100,
    log=True,
    sample=1,
    test_size=0.2,
    test_path=None,
    val_size=0.0,
    val_path=None,
    fine_tune=None,
    transfer=None,
    train=True,
    evaluate=True,
    optim="AdamW",
    learning_rate=3e-4,
    momentum=0.9,
    weight_decay=1e-6,
    batch_size=128,
    workers=0,
    device=torch.device("cpu"),
    **kwargs,
):
    assert (
        evaluate or train
    ), "No task given - Set at least one of 'train' or 'evaluate' kwargs as True"

    if test_path:
        test_size = 0.0

    if not (test_path and val_path):
        assert test_size + val_size < 1.0, (
            f"'test_size'({test_size}) "
            f"plus 'val_size'({val_size}) must be less than 1"
        )

    if ensemble > 1 and (fine_tune or transfer):
        raise NotImplementedError(
            "If training an ensemble with fine tuning or transfering"
            " options the models must be trained one by one using the"
            " run-id flag."
        )

    assert not (fine_tune and transfer), (
        "Cannot fine-tune and" " transfer checkpoint(s) at the same time."
    )

    model_dir = make_model_dir(model_name, ensemble, run_id)

    dataset = CompositionData(data_path=data_path, fea_path=fea_path, task=task)
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len

    train_idx = list(range(len(dataset)))

    if evaluate:
        if test_path:
            print(f"using independent test set: {test_path}")
            test_set = CompositionData(
                data_path=test_path, fea_path=fea_path, task=task
            )
            test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
        elif test_size == 0.0:
            raise ValueError("test-size must be non-zero to evaluate model")
        else:
            print(f"using {test_size} of training set as test set")
            train_idx, test_idx = split(
                train_idx, random_state=data_seed, test_size=test_size
            )
            test_set = torch.utils.data.Subset(dataset, test_idx)

    if train:
        if val_path:
            print(f"using independent validation set: {val_path}")
            val_set = CompositionData(data_path=val_path, fea_path=fea_path, task=task)
            val_set = torch.utils.data.Subset(val_set, range(len(val_set)))
        else:
            if val_size == 0.0 and evaluate:
                print("No validation set used, using test set for evaluation purposes")
                # NOTE that when using this option care must be taken not to
                # peak at the test-set. The only valid model to use is the one
                # obtained after the final epoch where the epoch count is
                # decided in advance of the experiment.
                val_set = test_set
            elif val_size == 0.0:
                val_set = None
            else:
                print(f"using {val_size} of training set as validation set")
                train_idx, val_idx = split(
                    train_idx,
                    random_state=data_seed,
                    test_size=val_size / (1 - test_size),
                )
                val_set = torch.utils.data.Subset(dataset, val_idx)

        train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])

    data_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }

    setup_params = {
        "loss": loss,
        "optim": optim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "device": device,
        "fine_tune": fine_tune,
        "transfer": transfer,
    }

    model_params = {
        "task": task,
        "robust": robust,
        "n_targets": n_targets,
        "elem_emb_len": elem_emb_len,
        "elem_fea_len": elem_fea_len,
        "n_graph": n_graph,
        "elem_heads": 3,
        "elem_gate": [256],
        "elem_msg": [256],
        "cry_heads": 3,
        "cry_gate": [256],
        "cry_msg": [256],
        "out_hidden": [1024, 512, 256, 128, 64],
    }

    if train:
        train_ensemble(
            model_class=Roost,
            model_dir=model_dir,
            ensemble_folds=ensemble,
            epochs=epochs,
            train_set=train_set,
            val_set=val_set,
            log=log,
            data_params=data_params,
            setup_params=setup_params,
            model_params=model_params,
        )

    if evaluate:

        data_reset = {
            "batch_size": 16 * batch_size,  # faster model inference
            "shuffle": False,  # need fixed data order due to ensembling
        }
        data_params.update(data_reset)

        results_func = (
            results_regression if task == "regression" else results_classification
        )
        results_func(
            model_class=Roost,
            model_dir=model_dir,
            ensemble_folds=ensemble,
            test_set=test_set,
            data_params=data_params,
            robust=robust,
            device=device,
            eval_type="checkpoint",
        )


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(
        description=(
            "Roost - a structure-agnostic message-passing "
            "neural network for inorganic materials"
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="roost",
        metavar="STR",
        help="Name for sub-directory where models will be stored",
    )

    # data inputs
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/datasets/expt-non-metals.csv",
        metavar="PATH",
        help="Path to main data set/training set",
    )

    # data embeddings
    parser.add_argument(
        "--fea-path",
        type=str,
        default="data/embeddings/matscholar-embedding.json",
        metavar="PATH",
        help="Element embedding feature path",
    )

    # graph inputs
    parser.add_argument(
        "--n-graph",
        default=3,
        type=int,
        metavar="INT",
        help="Number of message passing layers (default: 3)",
    )
    parser.add_argument(
        "--elem-fea-len",
        default=64,
        type=int,
        metavar="INT",
        help="Number of hidden features for elements (default: 64)",
    )

    return add_common_args(parser)


if __name__ == "__main__":
    args = input_parser()

    print(f"The model will run on the {args.device} device")

    main(**vars(args))
