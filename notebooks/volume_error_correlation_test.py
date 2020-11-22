# %%
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score

from roost.base import ROOT
from roost.plots import density_scatter, ptable_elemental_prevalence
from roost.plots.ptable import count_elements

# %% data paths
df_path = lambda idx: f"{ROOT}/models/mp-subset/ens_{idx}/test_results.csv"

dfs = {"Ens 0": df_path(0), "Ens 1": df_path(1), "Ens 2": df_path(2)}


# %% scatter plots
fig = plt.figure(figsize=(20, 7))
outer = gridspec.GridSpec(
    1, 3, wspace=0.25, hspace=0.15, left=0.10, right=0.95, bottom=0.15, top=0.99
)

for i, (title, f) in enumerate(dfs.items()):
    df = pd.read_csv(f, comment="#", na_filter=False)

    tar = df["target"].to_numpy()

    pred_cols = [col for col in df.columns if "pred" in col]
    pred = df[pred_cols].to_numpy().T
    mean = np.average(pred, axis=0)

    res = mean - tar
    mae = np.abs(res).mean()
    rmse = np.sqrt(np.square(res).mean())
    r2 = r2_score(tar, mean)

    gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer[i],
        width_ratios=(9, 2),
        height_ratios=(2, 9),
        wspace=0.0,
        hspace=0.0,
    )

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    ax_scatter.tick_params(direction="out")
    ax_histx.tick_params(direction="out", labelbottom=False)
    ax_histy.tick_params(direction="out", labelleft=False)

    # the scatter plot:
    density_scatter(tar, mean, ax_scatter)
    ax_scatter.set_xlabel("Target Value / eV", labelpad=8)
    ax_scatter.set_ylabel("Predicted Value / eV", labelpad=6)

    binwidth = 0.25
    top = np.ceil(np.array([tar, mean]).max() / binwidth) * binwidth
    bottom = (np.ceil(np.array([tar, mean]).min() / binwidth) - 1) * binwidth

    x_lims = np.array((bottom, top))
    y_lims = np.array((bottom, top))

    huber = HuberRegressor()
    huber.fit(tar.reshape(-1, 1), mean.reshape(-1, 1))
    ax_scatter.plot(x_lims, huber.predict(x_lims.reshape(-1, 1)), "r--")

    ax_scatter.plot(x_lims, y_lims - 0.05, color="grey", linestyle="--", alpha=0.3)
    ax_scatter.plot(x_lims, y_lims + 0.05, color="grey", linestyle="--", alpha=0.3)

    ax_scatter.set_xlim((x_lims))
    ax_scatter.set_ylim((y_lims))

    bins = np.arange(bottom, top + binwidth, binwidth)
    ax_histx.hist(tar, bins=bins, density=True)
    ax_histy.hist(mean, bins=bins, density=True, orientation="horizontal")

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    h_lim = max(ax_histy.get_xlim()[1], ax_histx.get_ylim()[1])
    ax_histx.set_ylim((0, h_lim))
    ax_histy.set_xlim((0, h_lim))
    ax_histx.axis("off")
    ax_histy.axis("off")

    ax_scatter.legend(title=r"$\bf{Model: {%s}}$" % (title), frameon=False, loc=2)

    ax_scatter.annotate(
        f"R2 = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
        (0.045, 0.7),
        xycoords="axes fraction",
    )
    plt.savefig(f"{ROOT}/models/mp-subset/pred-test-multi-wren.png")


# %%
# disable na_filter so the composition NaN is not converted to nan
dfs = [pd.read_csv(df_path(idx), na_filter=False) for idx in range(20)]
total = pd.concat(dfs)
# sanity check
# verify each material id appeared about (ensemble size x test set size) times
# in the test set, e.g. 20 x 0.2 = 4 (actual result: 4.049)
total.id.value_counts().mean()


# %%
total.to_csv(
    f"{ROOT}/models/mp-subset/combined_test_results.csv", index=False, float_format="%g"
)


# %%
total["ae"] = (total.target - total.pred_0).abs()


# %%
# total.plot.scatter(x="ae", y="pred_0")
# plt.savefig(f"{ROOT}/models/mp-subset/pred-vs-ae.png")
total.plot.scatter(x="ae", y="target")
plt.savefig(f"{ROOT}/models/mp-subset/target-vs-ae.png")


# %%
uniq = total.sort_values("ae").drop_duplicates("id", keep="last")
plt.plot(
    np.arange(0.05, 0.15, 0.01),
    [(uniq.ae > thr).mean() for thr in np.arange(0.05, 0.15, 0.01)],
)
plt.savefig(f"{ROOT}/models/mp-subset/uniq-ae-threshhold.png")


# %%
mean_per_id = total.groupby("id").mean()
df_mp = pd.read_csv(ROOT + "/data/datasets/large/mat_proj.csv", na_filter=False)

# add unit cell volume column to results dataframe
mean_per_id["volume"] = df_mp.loc[
    df_mp["material_id"].isin(mean_per_id.index)
].volume.values


# %%
mean_per_id.sort_values("volume", inplace=True)


# %%
mean_per_id.plot.scatter(
    x="pred_0", y="target", c=np.log(mean_per_id.volume), cmap="blues"
)
plt.savefig(f"{ROOT}/models/mp-subset/target-vs-pred_mean-per-id_color-by-volume.png")

