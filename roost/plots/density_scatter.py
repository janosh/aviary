import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score


def density_scatter(
    x, y, ax=None, colours=None, label=None, sort=True, log=True, bins=100, **kwargs
):
    """Scatter plot colored by 2d histogram"""
    if ax is None:
        ax = plt.gca()

    data, x_e, y_e = np.histogram2d(x, y, bins)

    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    if log:
        z = np.log(z)

    ax.scatter(x, y, c=z, cmap=colours or "Blues", label=label, **kwargs)
    return ax


def density_scatter_with_hists(targets, preds, fig, spec, title=None):
    mean = np.average(preds, axis=0)

    res = mean - targets
    mae = np.abs(res).mean()
    rmse = ((res ** 2).mean()) ** 0.5
    r2 = r2_score(targets, mean)

    gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
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
    density_scatter(targets, mean, ax_scatter)
    ax_scatter.set_xlabel("Target Value / eV", labelpad=8)
    ax_scatter.set_ylabel("Predicted Value / eV", labelpad=6)

    binwidth = 0.25
    top = np.ceil(np.array([targets, mean]).max() / binwidth) * binwidth
    bottom = (np.ceil(np.array([targets, mean]).min() / binwidth) - 1) * binwidth

    x_lims = np.array((bottom, top))
    y_lims = np.array((bottom, top))

    huber = HuberRegressor()
    huber.fit(targets.reshape(-1, 1), mean.reshape(-1, 1))
    ax_scatter.plot(x_lims, huber.predict(x_lims.reshape(-1, 1)), "r--")

    ax_scatter.plot(x_lims, y_lims - 0.05, color="grey", linestyle="--", alpha=0.3)
    ax_scatter.plot(x_lims, y_lims + 0.05, color="grey", linestyle="--", alpha=0.3)

    ax_scatter.set_xlim((x_lims))
    ax_scatter.set_ylim((y_lims))

    bins = np.arange(bottom, top + binwidth, binwidth)
    ax_histx.hist(targets, bins=bins, density=True)
    ax_histy.hist(mean, bins=bins, density=True, orientation="horizontal")

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    h_lim = max(ax_histy.get_xlim()[1], ax_histx.get_ylim()[1])
    ax_histx.set_ylim((0, h_lim))
    ax_histy.set_xlim((0, h_lim))
    ax_histx.axis("off")
    ax_histy.axis("off")

    if title:
        ax_scatter.legend(title=rf"$\bf{{Model: {title}}}$", frameon=False, loc=2)

    ax_scatter.annotate(
        f"R2 = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
        (0.045, 0.7),
        xycoords="axes fraction",
    )
