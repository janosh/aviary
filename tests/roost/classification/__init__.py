import torch
from sklearn.model_selection import train_test_split as split

from roost.core import ROOT
from roost.roost import CompositionData

task = "classification"

data_path = ROOT + "/data/datasets/bandgap-binary.csv"
fea_path = ROOT + "/data/embeddings/matscholar-embedding.json"

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
