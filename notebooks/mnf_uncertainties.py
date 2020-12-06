# %%
import matplotlib.pyplot as plt
import pandas as pd
import torch

from roost import plots
from roost.utils import ROOT

# %%
# model_name = "robust_mnf_mp"
model_name = "robust_mnf_oqmd"
run_id = "run_1"
model_dir = f"{ROOT}/models/{model_name}/{run_id}"

results = pd.read_csv(f"{model_dir}/test_results.csv")

# std column name can be  std_al_0, std_ep or std_tot
true, pred, std = results[["target", "pred_0", "std_tot"]].values.T

std_ep, std_al = results[["std_ep", "std_al_0"]].values.T


# %%
std_tot = ((std_ep * 200) ** 2 + std_al ** 2) ** 0.5
plots.err_decay(
    true,
    pred,
    {"std_ep": std_ep, "std_al": std_al, "std_tot": std_tot},
    percentile=False,
)
plots.err_decay(true, pred, std_ep, percentile=False)
# plt.savefig(f"{model_dir}/err_decay_by_std.pdf")

