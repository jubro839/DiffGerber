"""Close two gaps a reviewer would flag.

(a) The setup section promises a 2019-onward robustness check (the early-sample
    warehouse coverage gap) but the results never report it.
(b) The paper calls the Gerber statistic an extension of Kendall's tau but
    never benchmarks against rank correlations. Add Kendall and Spearman.
"""

import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats

torch.set_default_dtype(torch.float64)
RES = pathlib.Path(__file__).resolve().parent / "results"
WINDOW, STEP = 252, 21


def _shrink_to_identity(P, delta):
    """Linear shrinkage of a correlation matrix toward the identity."""
    return (1 - delta) * P + delta * np.eye(P.shape[0])


def _to_cov(P, R, delta):
    P = _shrink_to_identity(P, delta)
    s = R.std(0, ddof=1)
    return np.outer(s, s) * P


def spearman_cov(R, delta=0.0):
    """Rank correlation, rescaled to a covariance by the sample volatilities."""
    P = np.corrcoef(rankdata(R, axis=0), rowvar=False)
    return _to_cov(P, R, delta)


def kendall_cov(R, delta=0.0):
    """Kendall's tau matrix via the sign-covariance identity, mapped to a
    correlation by the elliptical relation rho = sin(pi*tau/2)."""
    n = R.shape[1]
    # tau = mean over pairs of sign(x_i - x_j) * sign(y_i - y_j); computed as a
    # Gram matrix of pairwise-difference signs, vectorized over assets.
    T = R.shape[0]
    idx = np.triu_indices(T, k=1)
    D = np.sign(R[idx[0], :] - R[idx[1], :])          # (pairs, n)
    tau = (D.T @ D) / D.shape[0]
    np.fill_diagonal(tau, 1.0)
    P = np.sin(np.pi * tau / 2.0)
    ev, V = np.linalg.eigh(P)                          # nearest PSD, unit diag
    P = V @ np.diag(np.clip(ev, 1e-8, None)) @ V.T
    d = np.sqrt(np.diag(P))
    P = P / np.outer(d, d)
    return _to_cov(P, R, delta)


mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index

rows_rank, rows_2019 = [], []
for topn, tag in [(100, "gs2022"), (400, "gs2022_top400")]:
    _m = {}
    def uni(d, topn=topn, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe(d, topn, px, mem)
        return _m[d]
    snap = json.loads((RES / f"scalar_thresholds_{tag}.json").read_text())
    cg = {int(k.split("_")[1]): math.log1p(math.exp(snap[k]["raw"]))
          for k in snap if k.startswith("tglobal_")}
    reb = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)]

    contenders = {
        "learned": lambda R, s_, d_: backtest.covariance_weight_fn(
            lambda X: baselines.gerber_cov(X, cg[d_.year], "gs2022"))(R, s_, d_),
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda X: baselines.gerber_cov(X, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(baselines.analytical_nonlinear_shrinkage),
        "spearman": backtest.covariance_weight_fn(spearman_cov),
        "kendall": backtest.covariance_weight_fn(kendall_cov),
        "spearman_shrunk": backtest.covariance_weight_fn(
            lambda X: spearman_cov(X, delta=0.2)),
        "kendall_shrunk": backtest.covariance_weight_fn(
            lambda X: kendall_cov(X, delta=0.2)),
    }
    runs = {}
    for start_year, bucket in [(2017, rows_rank), (2019, rows_2019)]:
        eval_start = min(d for d in reb if d.year >= start_year)
        for name, wfn in contenders.items():
            if start_year == 2019 and "spearman" in name or start_year == 2019 and "kendall" in name:
                continue
            t0 = time.time()
            r = backtest.run_walkforward(name, rets, wfn, uni, eval_start,
                                         dates[-1], window=WINDOW, step=STEP)
            m = r.metrics(cost_bps=10)
            bucket.append({"universe": f"top{topn}", "eval_from": start_year,
                           "method": name,
                           "ann_vol_net": round(m["ann_vol_net"], 5),
                           "sharpe_net": round(m["sharpe_net"], 3),
                           "n_folds": m["n_folds"]})
            if start_year == 2017:
                runs[name] = r
            print(f"  top{topn} {start_year}+ {name:12s} "
                  f"vol={m['ann_vol_net']*100:6.2f}%  sharpe={m['sharpe_net']:+.2f}  "
                  f"({m['n_folds']} folds, {time.time()-t0:.0f}s)", flush=True)
    for base in ["spearman", "kendall", "spearman_shrunk", "kendall_shrunk"]:
        dm, p = stats.diebold_mariano(runs["learned"].fold_losses(),
                                      runs[base].fold_losses())
        print(f"  top{topn} learned vs {base}: DM={dm:+.2f} p={p:.4f}", flush=True)

pd.DataFrame(rows_rank).to_csv(RES / "rank_baselines.csv", index=False)
pd.DataFrame(rows_2019).to_csv(RES / "eval_from_2019.csv", index=False)
print("\ndone.")
