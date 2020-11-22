# %%
import pandas as pd

from roost import plots
from roost.core import ROOT

# %%
model_name = "robust_mnf_expt"
run_id = "run_1"
model_dir = f"{ROOT}/models/{model_name}/{run_id}"

results = pd.read_csv(f"{model_dir}/test_results.csv")
# std column name can be  std_al_0, std_ep_repeat_50 or std_tot
true, std, pred = results[["target", "pred_0", "std_tot"]].values.T

# std_ep, std_al = results[["std_ep_repeat_50", "std_al_0"]].values.T
# std_tot = ((std_ep * 1000) ** 2 + std_al ** 2) ** 0.5

plots.err_decay(true, pred, std)
