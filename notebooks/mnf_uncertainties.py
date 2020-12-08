# %%
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
import torch

from roost import plots
from roost.utils import ROOT

# %%
# model_name = "robust_mnf_mp"
# model_name = "robust_mnf_oqmd"
model_name = "robust_mnf_expt"
run_id = "run_1"
model_dir = f"{ROOT}/models/{model_name}/{run_id}"

df = pd.read_csv(f"{model_dir}/test_results.csv")

# std column name can be  std_al_0, std_ep or std_tot
true, pred, std, std_ep, std_al = df[
    ["target", "pred_0", "std_tot", "std_ep", "std_al_0"]
].values.T


# %%
std_tot = (std_ep ** 2 + std_al ** 2) ** 0.5
plots.err_decay(true, pred, {"std_ep": std_ep, "std_al": std_al, "std_tot": std_tot})
plt.savefig(f"{model_dir}/err_decay_by_std.pdf")


# %%
checkpoint = torch.load(model_dir + "/checkpoint.pth.tar", map_location="cpu")
checkpoint["epoch"]


# %%
webbrowser.open(f"file://{model_dir}")


# %%
plots.std_calibration(
    true, pred, {"std_ep": std_ep, "std_al": std_al, "std_tot": std_tot}
)
