import torch

from roost.roost import collate_batch

torch.manual_seed(0)  # ensure reproducible results


# minimal set of required Roost parameters

data_params_train = {
    "batch_size": 128,
    "collate_fn": collate_batch,
}

data_params_test = {
    "batch_size": 64 * 128,  # faster model inference
    "collate_fn": collate_batch,
    "shuffle": False,  # need fixed data order when ensembling
}

setup_params = {
    "loss": "L1",
    "optim": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-6,
    "momentum": 0.9,
    "device": torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu"),
}
