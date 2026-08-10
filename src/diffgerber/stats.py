"""Statistical tests: Diebold-Mariano and stationary bootstrap."""

import numpy as np


def auto_lag(n):
    """Newey-West automatic bandwidth: floor(4 (n/100)^(2/9))."""
    return max(1, int(4 * (n / 100) ** (2 / 9)))


def diebold_mariano(loss_a, loss_b, h=None, weights="bartlett"):
    """DM test on paired loss series (e.g., per-fold realized variance).

    Returns (dm_stat, p_value); negative stat means A has lower loss.

    h: number of lags + 1. h=1 applies NO autocorrelation correction. The
    folds are non-overlapping 21-day realized variances, so the MA(h-1)
    argument permits h=1, but the loss differentials are persistent
    (rho_1 ~ 0.32 for some pairs), which understates the variance and
    overstates significance. Default is the Newey-West automatic bandwidth.

    weights: 'bartlett' (Newey-West, guarantees a non-negative variance) or
    'rectangular' (the DM 1995 / HLN 1997 truncated sum).
    """
    from scipy import stats as sps

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if h is None:
        h = auto_lag(n)
    dbar = d.mean()
    gamma0 = ((d - dbar) ** 2).mean()
    var = gamma0
    for lag in range(1, h):
        cov = ((d[lag:] - dbar) * (d[:-lag] - dbar)).mean()
        w = (1 - lag / h) if weights == "bartlett" else 1.0
        var += 2 * w * cov
    var = max(var, 1e-300)
    dm = dbar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2 * sps.t.sf(abs(dm), df=n - 1)
    return dm, p


def stationary_bootstrap(x, stat_fn, n_boot=2000, mean_block=21, seed=0):
    """Politis-Romano stationary bootstrap CI for stat_fn of a series."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    p = 1.0 / mean_block
    stats_ = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        stats_[b] = stat_fn(x[idx])
    return np.percentile(stats_, [2.5, 50, 97.5])
