# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ml_matrics import (
    count_elements,
    cum_err_cum_res,
    density_scatter_hex_with_hist,
    density_scatter_with_hist,
    ptable_elemental_prevalence,
)
from scipy.stats import zscore
from sklearn.metrics import r2_score

from aviary.utils import ROOT


plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14

log_color_scale = mpl.colors.LogNorm()


# %%
df_path = lambda idx: f"{ROOT}/models/mp_subset/ens_{idx}/test_results.csv"

# disable na_filter so the composition NaN is not converted to nan
dfs = [pd.read_csv(df_path(idx), na_filter=False) for idx in range(20)]
total = pd.concat(dfs).set_index("id")

# sanity check
# verify each material id appeared about (ensemble size x test set size) times
# in the test set, e.g. 20 x 0.2 = 4 (actual result: 4.049)
total.index.value_counts().mean()
total["res"] = total.target - total.pred_0
total["ae"] = (total.target - total.pred_0).abs()


# %%
df_mp = pd.read_csv(ROOT + "/data/datasets/large/mat_proj.csv", na_filter=False)
df_mp = df_mp.rename(columns={"material_id": "id"}).set_index("id")


# %%
for key in [
    "composition",
    "formation_energy_per_atom",
    "e_above_hull",
    "band_gap",
    "nelements",
    "nsites",
    "volume",
    "spacegroup",
]:
    total[key] = df_mp[key]


# %%
# total.to_csv(f"{ROOT}/models/mp_subset/combined_test_results.csv", float_format="%g")
total = pd.read_csv(
    f"{ROOT}/models/mp_subset/combined_test_results.csv", index_col="id"
)


# %% data paths
dfs = [pd.read_csv(df_path(idx), comment="#", na_filter=False) for idx in range(3)]


# %%
fig = plt.figure(figsize=(21, 7))
outer_grid = fig.add_gridspec(1, 3)
for df, cell in zip(dfs, outer_grid):

    res = df.pred_0 - df.target
    mae = np.abs(res).mean()
    rmse = ((res ** 2).mean()) ** 0.5
    r2 = r2_score(df.target, df.pred_0)

    density_scatter_with_hist(
        df.target,
        df.pred_0,
        cell,
        xlabel="Target Enthalpy [eV/atom]",
        ylabel="Predicted Enthalpy [eV/atom]",
        text=f"R2 = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
    )

plt.savefig(f"{ROOT}/models/mp_subset/density-scatter.png")


# %%
total.sort_values("volume").plot.scatter(
    x="ae",
    y="target",  # or "pred_0"
    c="volume",
    cmap="Blues",
    norm=log_color_scale,
    title="sort volume asc",
)
plt.savefig(f"{ROOT}/models/mp_subset/target-vs-ae-sort-volume-asc.png")


# %%
# inverted cumulative error plot (useless)
uniq = total.sort_values("ae").drop_duplicates("id", keep="last")
plt.plot(
    np.arange(0.05, 0.15, 0.01),
    [(uniq.ae > thr).mean() for thr in np.arange(0.05, 0.15, 0.01)],
)
plt.savefig(f"{ROOT}/models/mp_subset/uniq-ae-threshhold.png")


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
plt.savefig(f"{ROOT}/models/mp_subset/target-vs-pred_total_color-by-volume.png")


# %%
mean_per_id.sort_values("volume").plot.scatter(
    x="pred_0",
    y="target",
    c="volume",
    cmap="Blues",
    norm=log_color_scale,
)
plt.savefig(f"{ROOT}/models/mp_subset/target-vs-pred_mean-per-id_color-by-volume.png")


# %%
high_err = mean_per_id.sort_values("ae").tail(len(mean_per_id) // 5)
low_err = mean_per_id.sort_values("ae").head(len(mean_per_id) * 4 // 5)


# %%
ptable_elemental_prevalence(high_err.composition.values, log_scale=True)
plt.savefig(f"{ROOT}/models/mp_subset/ptable_count_by_err.png")


# %%
high_err_counts = count_elements(high_err.composition.values)
low_err_counts = count_elements(low_err.composition.values)


# %%
low_err_counts[low_err_counts == 0] = 1
relative_counts = high_err_counts / low_err_counts
ptable_elemental_prevalence(elem_counts=relative_counts)
plt.savefig(f"{ROOT}/models/mp_subset/relative_ptable_count_by_err.png")


# %%
targets, preds = total[["target", "pred_0"]].values.T

res = preds - targets
mae = np.abs(res).mean()
rmse = np.sqrt(np.square(res).mean())
r2 = r2_score(targets, preds)

title = r"$\bf{Model: Wren}$"


# %%
density_scatter_hex_with_hist(
    targets,
    preds,
    title=title,
    text=f"R2 = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
    color_by=total.volume,
)

plt.savefig(f"{ROOT}/models/mp_subset/hex-pred-vs-target-color-by-volume-log-mean.png")


# %%
cum_err_cum_res(targets, preds, [title])

plt.savefig(f"{ROOT}/models/mp_subset/cum_err_cum_res.png")


# %%
top_spacegroups = total.value_counts("spacegroup").head(20).index
no_outliers = total[np.abs(zscore(total.res)) < 0.5]


# %%
sns.violinplot(
    data=no_outliers.loc[lambda x: x.spacegroup.isin(top_spacegroups)],
    x="spacegroup",
    y="res",
    scale="count",
    cut=0,
)
plt.savefig(f"{ROOT}/models/mp_subset/violin-ae-vs-spacegroup.png")


# %%
# remove outliers (https://stackoverflow.com/a/23202269)
total[np.abs(zscore(total.res)) < 0.1]


# %%
# the spacegroups with highest MAE and their number of samples
top_mae_spacegroups = pd.DataFrame(
    total.groupby("spacegroup").ae.mean().sort_values(ascending=False).head(10)
).rename(columns={"ae": "mae"})
top_mae_spacegroups["count"] = pd.DataFrame(total.value_counts("spacegroup"))

top_mae_spacegroups.plot(kind="bar", y=["mae", "count"], secondary_y="count")
plt.savefig(f"{ROOT}/models/mp_subset/bar-mae+count-vs-spacegroup.pdf")
