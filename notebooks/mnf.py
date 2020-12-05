# %%
from argparse import ArgumentParser

import torch
from sklearn.model_selection import train_test_split as split

from roost.roost import CompositionData, Roost, collate_batch
from roost.utils import ROOT, make_model_dir, run_test, train_ensemble

torch.manual_seed(0)  # ensure reproducible results

# %%
fea_path = ROOT + "/data/embeddings/matscholar-embedding.json"
task = "regression"
loss = "L1"
robust = True
ensemble = 1
data_seed = 0
log = True
sample = 1
test_size = 0.2
optim = "AdamW"
learning_rate = 3e-4
momentum = 0.9
weight_decay = 1e-6
batch_size = 128
test_repeat = 30
verbose = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Now running on {device}")

parser = ArgumentParser(allow_abbrev=False)
parser.add_argument("--model-name", type=str, default="robust_mnf_oqmd")
parser.add_argument("--run-id", type=str, default="run_1")
parser.add_argument("--use-mnf", action="store_false")  # False by default
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument(
    "--data-path", type=str, default=f"{ROOT}/data/datasets/oqmd-form-enthalpy.csv"
)
flags, _ = parser.parse_known_args()

args = ["model_name", "run_id", "use_mnf", "epochs", "data_path"]
model_name, run_id, use_mnf, epochs, data_path = [vars(flags).get(x) for x in args]

model_dir = make_model_dir(model_name, ensemble, run_id)


# %%
dataset = CompositionData(data_path, fea_path, task)

train_idx = list(range(len(dataset)))


# %% Testing
print(f"Using {test_size} of training set as test set")
train_idx, test_idx = split(train_idx, random_state=data_seed, test_size=test_size)
test_set = torch.utils.data.Subset(dataset, test_idx)


# %%
print("No validation set used, using test set for evaluation purposes")
# NOTE that when using this option care must be taken not to
# peak at the test-set. The only valid model to use is the one
# obtained after the final epoch where the epoch count is
# decided in advance of the experiment.

train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])

data_params = {
    "batch_size": batch_size,
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
}

model_params = {
    "task": task,
    "robust": robust,
    "n_targets": dataset.n_targets,
    "elem_emb_len": dataset.elem_emb_len,
    "elem_heads": 3,
    "elem_gate": [256],
    "elem_msg": [256],
    "cry_heads": 3,
    "cry_gate": [256],
    "cry_msg": [256],
    "out_hidden": [1024, 512, 256, 128, 64],
    "use_mnf": use_mnf,
}


# %% Train a Roost model
train_ensemble(
    model_class=Roost,
    model_dir=model_dir,
    ensemble_folds=ensemble,
    epochs=epochs,
    train_set=train_set,
    val_set=test_set,
    log=log,
    verbose=verbose,
    data_params=data_params,
    setup_params=setup_params,
    model_params=model_params,
)


# %% Evaluate a Roost model
data_params["batch_size"] = 64 * batch_size  # faster model inference
data_params["shuffle"] = False  # need fixed data order due to ensembling

run_test(
    task,
    model_class=Roost,
    model_dir=model_dir,
    ensemble_folds=ensemble,
    test_set=test_set,
    data_params=data_params,
    robust=robust,
    eval_type="checkpoint",
    device=device,
    repeat=test_repeat,
)
