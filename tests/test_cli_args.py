from argparse import ArgumentParser

from examples.common_cli_args import add_common_args

common_args = [
    "val_path",
    "val_size",
    "test_path",
    "test_size",
    "workers",
    "batch_size",
    "data_seed",
    "sample",
    "epochs",
    "loss",
    "robust",
    "optim",
    "learning_rate",
    "momentum",
    "weight_decay",
    "ensemble",
    "run_id",
    "fine_tune",
    "transfer",
    "task",
    "evaluate",
    "train",
    "disable_cuda",
    "log",
]


def test_common_cli_args():
    parser = ArgumentParser()
    args = add_common_args(parser)
    for arg in common_args:
        assert hasattr(args, arg)
