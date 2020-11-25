# %%
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from roost.base import ROOT
from roost.plots import (
    count_elements,
    density_scatter_with_hists,
    pred_target_hex_scatter,
    ptable_elemental_prevalence,
)

plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14

log_color_scale = mpl.colors.LogNorm()


# %% data paths
df_path = lambda idx: f"{ROOT}/models/mp-subset/ens_{idx}/test_results.csv"

dfs = {"Ens 0": df_path(0), "Ens 1": df_path(1), "Ens 2": df_path(2)}


# %% scatter plots
fig = plt.figure(figsize=(20, 7))
outer = gridspec.GridSpec(
    1, 3, wspace=0.25, hspace=0.15, left=0.10, right=0.95, bottom=0.15, top=0.99
)

for i, (title, file_path) in enumerate(dfs.items()):
    df = pd.read_csv(file_path, comment="#", na_filter=False)

    targets = df["target"].to_numpy()

    pred_cols = [col for col in df.columns if "pred" in col]
    preds = df[pred_cols].to_numpy().T
    density_scatter_with_hists(targets, preds, fig, outer[i])
    plt.savefig(f"{ROOT}/models/mp-subset/pred-test-multi-wren.png")


# %%
# disable na_filter so the composition NaN is not converted to nan
dfs = [pd.read_csv(df_path(idx), na_filter=False) for idx in range(20)]
total = pd.concat(dfs).set_index("id")
# sanity check
# verify each material id appeared about (ensemble size x test set size) times
# in the test set, e.g. 20 x 0.2 = 4 (actual result: 4.049)
total.index.value_counts().mean()
total["ae"] = (total.target - total.pred_0).abs()


# %%
df_mp = pd.read_csv(ROOT + "/data/datasets/large/mat_proj.csv", na_filter=False)
df_mp = df_mp.rename(columns={"material_id": "id"}).set_index("id")
# add crystal unit cell volume to results dataframe
total["volume"] = df_mp.volume
total["composition"] = df_mp.composition


# %%
for key in ["volume", "composition", "e_above_hull", "band_gap", "nsites"]:
    total[key] = df_mp[key]


# %%
# total.to_csv(f"{ROOT}/models/mp-subset/combined_test_results.csv", float_format="%g")
total = pd.read_csv(f"{ROOT}/models/mp-subset/combined_test_results.csv")


# %%
total.sort_values("volume").plot.scatter(
    x="ae",
    y="target",  # or "pred_0"
    c="volume",
    cmap="Blues",
    norm=log_color_scale,
    title="sort volume asc",
)
plt.savefig(f"{ROOT}/models/mp-subset/target-vs-ae-sort-volume-asc.png")


# %%
# inverted cumulative error plot (useless)
uniq = total.sort_values("ae").drop_duplicates("id", keep="last")
plt.plot(
    np.arange(0.05, 0.15, 0.01),
    [(uniq.ae > thr).mean() for thr in np.arange(0.05, 0.15, 0.01)],
)
plt.savefig(f"{ROOT}/models/mp-subset/uniq-ae-threshhold.png")


# %%
mean_per_id = total.groupby("id").mean()
# mean drops string columns, need to reinstate composition
mean_per_id["composition"] = df_mp.composition


# %%
total.sort_values("volume").plot.scatter(
    x="pred_0",
    y="target",
    c="volume",
    cmap="Blues",
    norm=log_color_scale,
)
plt.savefig(f"{ROOT}/models/mp-subset/target-vs-pred_total_color-by-volume.png")


# %%
mean_per_id.sort_values("volume").plot.scatter(
    x="pred_0",
    y="target",
    c="volume",
    cmap="Blues",
    norm=log_color_scale,
)
plt.savefig(f"{ROOT}/models/mp-subset/target-vs-pred_mean-per-id_color-by-volume.png")


# %%
high_err = mean_per_id.sort_values("ae").tail(len(mean_per_id) // 5)
low_err = mean_per_id.sort_values("ae").head(len(mean_per_id) * 4 // 5)


# %%
ptable_elemental_prevalence(high_err.composition.values, log_scale=True)
plt.savefig(f"{ROOT}/models/mp-subset/ptable_count_by_err.png")


# %%
high_err_counts = count_elements(high_err.composition.values)
low_err_counts = count_elements(low_err.composition.values)


# %%
low_err_counts[low_err_counts == 0] = 1
relative_counts = high_err_counts / low_err_counts
ptable_elemental_prevalence(elem_counts=relative_counts)
plt.savefig(f"{ROOT}/models/mp-subset/relative_ptable_count_by_err.png")


# %%
targets, preds = total[["target", "pred_0"]].values.T

res = preds - targets
mae = np.abs(res).mean()
rmse = np.sqrt(np.square(res).mean())
r2 = r2_score(targets, preds)


# %%
title = r"$\bf{Model: Wren}$"
text = f"R2 = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}"
pred_target_hex_scatter(targets, preds, title, text, color_by=total.volume)

plt.savefig(f"{ROOT}/models/mp-subset/hex-pred-vs-target-color-by-volume-log-mean.png")
