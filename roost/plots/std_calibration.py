import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .utils import add_identity


def std_calibration(true, pred, stds):
    # Inspired by https://github.com/Ryan-Rhys/FlowMO#uncertainty-calibration
    if type(stds) != dict:
        stds = {"std": stds}

    res = np.abs(pred - true)

    xs = np.linspace(0, 1, 1000)

    add_identity(label="ideal")
    for key, std in stds.items():
        np.random.shuffle(std)
        z = res / std
        cali = np.zeros_like(xs)

        for i, val in enumerate(xs):
            # ppf: Percent Point Function (inverse of CDF)
            q_test = norm.ppf((val + 1) / 2)
            cali[i] = np.mean(z < q_test)

        plt.plot(xs, cali, label=key)

    plt.legend()
    plt.show()
