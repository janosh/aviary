import numpy as np
from matplotlib import pyplot as plt


def get_err_decay(y_test, y_pred, y_std, n_rand=50):
    abs_err = np.abs(y_test - y_pred)
    y_std_sort = np.argsort(y_std)  # indices that sort y_std in ascending uncertainty
    n_inc = range(1, len(abs_err) + 1)  # numbers of samples in cumsum()

    decay_by_std = abs_err[y_std_sort].cumsum() / n_inc
    decay_by_err = np.sort(abs_err).cumsum() / n_inc

    # error decay for random exclusion of samples
    ae_tile = np.tile(abs_err, [n_rand, 1])
    [np.random.shuffle(row) for row in ae_tile]  # shuffle rows of ae_tile in place
    rand_mean = ae_tile.cumsum(1).mean(0) / n_inc
    rand_std = ae_tile.cumsum(1).std(0) / n_inc

    return decay_by_std, decay_by_err, rand_mean, rand_std


def err_decay(y_test, y_pred, y_std, title=None, n_rand=50):
    """Plot for assessing the quality of uncertainty estimates. If a model's
    uncertainty is well calibrated, i.e. strongly correlated with its error,
    removing the most uncertain predictions should make the mean error decay
    similarly to how it decays when removing the predictions of largest error
    """

    decay_by_std, decay_by_err, rand_mean, rand_std = get_err_decay(
        y_test, y_pred, y_std, n_rand=n_rand
    )

    countdown = range(len(y_test), 0, -1)
    plt.plot(countdown, decay_by_std)
    plt.plot(countdown, decay_by_err)
    plt.plot(countdown, rand_mean)
    plt.fill_between(countdown, rand_mean + rand_std, rand_mean - rand_std, alpha=0.2)
    plt.ylim([0, decay_by_err[-1] * 1.1])

    # n: Number of remaining points in err calculation after discarding the
    # (len(y_test) - n) most uncertain/hightest-error points
    plt.xlabel("$n$")
    plt.ylabel("$\\epsilon_\\mathrm{MAE}$")
    plt.title(title)

    fig = plt.gcf()
    plt.show()
    return fig
