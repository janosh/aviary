import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn


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
