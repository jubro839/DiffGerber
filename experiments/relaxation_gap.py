"""How far is the relaxation from the statistic it relaxes?

The straight-through construction only earns its place if deploying the soft
Gerber matrix instead of the hard one would actually change the portfolio.
This measures that gap directly, at every evaluation fold and at each
temperature on the annealing path, in two units:

    correlation gap   ||G_soft - G_hard||_F / ||G_hard||_F
    decision gap      ||w_soft - w_hard||_1 / ||w_hard||_1

The second is the one that matters. A relaxation can sit close to the hard
matrix and still move a minimum-variance solution a long way, because the
optimizer inverts it; reporting only the matrix norm would understate the
case for the straight-through estimator.

No training here: both matrices are evaluated at the same fixed threshold, so
this isolates the relaxation from the learning.
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, data
from diffgerber.soft_gerber import hard_gerber, soft_exceedance, gerber_from_ternary

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
TOP_N = int(os.environ.get("DG_TOPN", 100))
TAUS = [0.2, 0.1, 0.05, 0.02]          # the annealing path, coarse to sharp
C = float(os.environ.get("DG_C", 0.5))
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index

_m = {}
def uni(d):
    if d not in _m:
        _m[d] = data.pit_universe(d, TOP_N, px, mem)
    return _m[d]


def soft_gerber_matrix(z, c, tau):
    """The soft matrix as it would be deployed, i.e. with no hard override."""
    u, d, n = soft_exceedance(z, c, tau)
    return gerber_from_ternary(u - d, n=n, normalization="gs2022")


rows = []
# same fold set the backtest reports on: every fold has a full 21-day
# realization horizon inside the sample, so this is the 114 evaluation
# rebalances of the paper, not one more
folds = [dates[i] for i in range(WINDOW, len(dates) - STEP, STEP)
         if dates[i].year >= 2017]
print(f"{len(folds)} evaluation folds, top-{TOP_N}, c={C}")

for tau in TAUS:
    corr_gaps, w_gaps = [], []
    for d0 in folds:
        i = dates.get_loc(d0)
        syms = uni(d0)
        R = torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values)
        s = R.std(0, unbiased=True).clamp_min(1e-8)
        z = R / s
        G_hard = hard_gerber(z, C, "gs2022")
        G_soft = soft_gerber_matrix(z, C, tau)
        corr_gaps.append(float(torch.linalg.norm(G_soft - G_hard)
                               / torch.linalg.norm(G_hard)))
        cov = lambda G: (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy()
        w_h = backtest.gmv_from_cov(cov(G_hard))
        w_s = backtest.gmv_from_cov(cov(G_soft))
        w_gaps.append(float(np.abs(w_s - w_h).sum() / np.abs(w_h).sum()))
    rows.append({"universe": f"top{TOP_N}", "c": C, "tau": tau,
                 "n_folds": len(folds),
                 "corr_gap_frobenius_mean": round(float(np.mean(corr_gaps)), 5),
                 "corr_gap_frobenius_max": round(float(np.max(corr_gaps)), 5),
                 "weight_gap_l1_mean": round(float(np.mean(w_gaps)), 5),
                 "weight_gap_l1_max": round(float(np.max(w_gaps)), 5)})
    print(f"  tau={tau:<5} correlation {np.mean(corr_gaps)*100:5.1f}%   "
          f"decision {np.mean(w_gaps)*100:5.1f}% of gross notional "
          f"(max {np.max(w_gaps)*100:.1f}%)", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / f"relaxation_gap_top{TOP_N}.csv", index=False)
print("\nrelaxation gap done.")
