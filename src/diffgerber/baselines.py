"""Classical covariance estimators (numpy in, numpy out)."""

import numpy as np

from .soft_gerber import hard_gerber


def _to_cov(corr, std):
    return np.outer(std, std) * corr


def sample_cov(R):
    """R: (T, N) returns array."""
    return np.cov(R, rowvar=False, ddof=1)


def ewma_cov(R, halflife=126):
    T = R.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    Rc = R - R.mean(0)
    return (Rc * w[:, None]).T @ Rc


def ledoit_wolf_cov(R):
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(R).covariance_


def oas_cov(R):
    from sklearn.covariance import OAS

    return OAS().fit(R).covariance_


def analytical_nonlinear_shrinkage(R):
    """Ledoit-Wolf (2020) analytical nonlinear shrinkage (ANS).

    Direct port of the published analytical_shrinkage.m, including the
    p > n branch. R: (T, N) returns.
    """
    X = np.asarray(R, dtype=float)
    X = X - X.mean(0)
    T, p = X.shape
    n = T - 1
    S = X.T @ X / n
    lam, U = np.linalg.eigh(S)
    lam = lam[max(0, p - n):]                # nonzero sample eigenvalues
    L = np.tile(lam, (min(p, n), 1)).T       # (m, m) with rows lambda_i
    h = n ** (-1 / 3)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3 / 4 / np.sqrt(5)) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hftemp = (-3 / 10 / np.pi) * x + (3 / 4 / np.sqrt(5) / np.pi) * (
            1 - x**2 / 5) * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    Hftemp[np.abs(x) == np.sqrt(5)] = ((-3 / 10 / np.pi) * x)[np.abs(x) == np.sqrt(5)]
    Hftemp[~np.isfinite(Hftemp)] = 0.0
    Hftilde = np.mean(Hftemp / H, axis=1)
    if p <= n:
        dtilde = lam / ((np.pi * (p / n) * lam * ftilde) ** 2
                        + (1 - p / n - np.pi * (p / n) * lam * Hftilde) ** 2)
        dall = dtilde
    else:
        Hftilde0 = (1 / np.pi) * (3 / 10 / h**2 + 3 / 4 / np.sqrt(5) / h
                    * (1 - 1 / 5 / h**2)
                    * np.log((1 + np.sqrt(5) * h) / abs(1 - np.sqrt(5) * h))
                    ) * np.mean(1 / lam)
        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lam / (np.pi**2 * lam**2 * (ftilde**2 + Hftilde**2))
        dall = np.concatenate([np.full(p - n, dtilde0), dtilde1])
    return U @ np.diag(dall) @ U.T


def gerber_cov(R, c=0.5, normalization="cos"):
    """Hard Gerber covariance at threshold c (torch under the hood)."""
    import torch

    Rt = torch.as_tensor(R, dtype=torch.float64)
    s = Rt.std(0, unbiased=True)
    G = hard_gerber(Rt / s, c, normalization)
    return (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy()


def hrp_weights(R, corr=None, linkage_method="single"):
    """Hierarchical Risk Parity (Lopez de Prado) — returns weights directly.

    corr: optional correlation matrix (ndarray) to drive BOTH the clustering
    distance and the covariance used for cluster variances (rescaled by sample
    volatilities). With corr=None the behaviour is the original HRP on the
    sample correlation, verified bit-identical to the reference algorithm.
    linkage_method: scipy linkage ('single' = original HRP; 'average'/'ward'
    are literature variants that mitigate single-linkage chaining).
    """
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    import pandas as pd

    if corr is None:
        cov = pd.DataFrame(sample_cov(R))
        corr = pd.DataFrame(np.corrcoef(R, rowvar=False))
    else:
        corr = pd.DataFrame(np.asarray(corr))
        s = R.std(0, ddof=1)
        cov = pd.DataFrame(np.outer(s, s) * corr.values)
    dist = np.sqrt(0.5 * (1 - corr)).values
    link = sch.linkage(ssd.squareform(dist, checks=False), method=linkage_method)
    sort_ix = sch.leaves_list(sch.optimal_leaf_ordering(link, ssd.squareform(dist, checks=False)))

    def cluster_var(items):
        sub = cov.iloc[items, items].values
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return ivp @ sub @ ivp

    w = pd.Series(1.0, index=sort_ix)
    clusters = [list(sort_ix)]
    while clusters:
        clusters = [
            c[j:k]
            for c in clusters
            for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                continue
            left, right = clusters[i], clusters[i + 1]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return w.sort_index().values


ESTIMATORS = {
    "sample": sample_cov,
    "ewma": ewma_cov,
    "ledoit_wolf": ledoit_wolf_cov,
    "oas": oas_cov,
    "gerber_cos_0.5": lambda R: gerber_cov(R, 0.5, "cos"),
    "gerber_2022_0.5": lambda R: gerber_cov(R, 0.5, "gs2022"),
}
