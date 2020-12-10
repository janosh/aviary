from os.path import isfile

import torch
from sklearn.model_selection import train_test_split as split

from roost.utils import ROOT

setup_params = {
    "loss": "L1",
    "optim": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-6,
    "momentum": 0.9,
    "device": "cpu",
}


def get_data(data_class, task, data_file, emb_file):

    fea_path = f"{ROOT}/data/embeddings/{emb_file}"
    data_path = f"{ROOT}/data/datasets/{data_file}"

    dataset = data_class(data_path, fea_path, task)

    train_idx = range(len(dataset))
    train_idx, test_idx = split(train_idx, random_state=0, test_size=0.2)

    test_set = torch.utils.data.Subset(dataset, test_idx)
    train_set = torch.utils.data.Subset(dataset, train_idx)

    model_params = {
        key: getattr(dataset, key)
        for key in ["n_targets", "elem_emb_len", "nbr_fea_len"]
        if hasattr(dataset, key)
    }

    return train_set, test_set, model_params


def get_params(model_class, task, data_class, collate_fn, data_file, emb_file):

    train_set, test_set, model_params = get_data(data_class, task, data_file, emb_file)

    model_params = {"task": task, "robust": True, **model_params}

    train_kwargs = {
        "model_class": model_class,
        "epochs": 1,
        "train_set": train_set,
        "val_set": test_set,
        "log": False,
        "verbose": False,
        "data_params": {
            "batch_size": 128,
            "collate_fn": collate_fn,
        },
        "setup_params": setup_params,
        "model_params": model_params,
    }

    data_test_params = {
        "batch_size": 64 * 128,  # faster model inference
        "collate_fn": collate_fn,
        "shuffle": False,  # need fixed data order when ensembling
    }

    return train_kwargs, data_test_params, test_set


# def assert_output_files(model_dir):
#     assert isfile(model_dir + '/checkpoint.pth.tar'), ''
