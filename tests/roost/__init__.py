import torch
from sklearn.model_selection import train_test_split as split

from roost.core import ROOT
from roost.roost import CompositionData, Roost
from tests import data_params_test, data_params_train, setup_params


def get_params_data(task, dataset):

    fea_path = ROOT + "/data/embeddings/matscholar-embedding.json"
    data_path = f"{ROOT}/data/datasets/{dataset}"
    dataset = CompositionData(data_path, fea_path, task)

    model_params = {
        "task": task,
        "robust": True,
        "n_targets": dataset.n_targets,
        "elem_emb_len": dataset.elem_emb_len,
    }

    train_idx = range(len(dataset))
    train_idx, test_idx = split(train_idx, random_state=0, test_size=0.2)

    test_set = torch.utils.data.Subset(dataset, test_idx)
    train_set = torch.utils.data.Subset(dataset, train_idx)

    train_kwargs = {
        "model_class": Roost,
        "epochs": 1,
        "train_set": train_set,
        "val_set": test_set,
        "log": False,
        "verbose": False,
        "data_params": data_params_train,
        "setup_params": setup_params,
        "model_params": model_params,
    }

    return train_kwargs, test_set
