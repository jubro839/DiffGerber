"""Threshold-sensitivity grid: OOS performance of hard GS2022 Gerber at fixed
thresholds c in {0.25..2.25}, vs the decision-learned per-year thresholds.

Answers two questions for the paper:
1. How threshold-sensitive is the Gerber estimator OOS? (figure data)
2. Does decision-learning recover the ex-post-optimal threshold without
   oracle access? (headline framing)
"""

import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffgerber import backtest, baselines, data, stats

torch.set_default_dtype(torch.float64)

NORM = "gs2022"
START_DATA, END_DATA = "2015-01-01", "2026-07-28"
WINDOW, STEP = 252, 21
TOP_N = int(os.environ.get("DG_TOPN", 100))
TAG = NORM + (f"_top{TOP_N}" if TOP_N != 100 else "")
EVAL_START_YEAR = 2017
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

print("loading data...")
mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return(START_DATA, END_DATA, symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index

_uni_memo = {}
def universe_fn(as_of):
    if as_of not in _uni_memo:
        _uni_memo[as_of] = data.pit_universe(as_of, TOP_N, px, mem)
    return _uni_memo[as_of]

reb_idx = list(range(WINDOW, len(dates) - 1, STEP))
fold_dates = [dates[i] for i in reb_idx]
eval_start = min(d for d in fold_dates if d.year >= EVAL_START_YEAR)

# learned per-year thresholds from the Phase 4 run
snap = json.loads((RESULTS / f"scalar_thresholds_{TAG}.json").read_text())
learned_c = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
             for k in snap if k.startswith("tglobal_")}
print("learned per-year c:", {y: round(c, 3) for y, c in sorted(learned_c.items())})

def fixed_fn(c):
    return backtest.covariance_weight_fn(lambda R: baselines.gerber_cov(R, c, NORM))

def learned_fn(R, symbols, as_of):
    return backtest.covariance_weight_fn(
        lambda R_: baselines.gerber_cov(R_, learned_c[as_of.year], NORM)
    )(R, symbols, as_of)

# validation-selected c: per eval year, pick the grid c minimizing mean fold
# loss over all folds strictly before that year (no gradient, no oracle)
print("computing per-fold losses for validation selection...")
full_start = fold_dates[0]
full_losses = {}
for c in GRID:
    r = backtest.run_walkforward(f"full_c{c}", rets, fixed_fn(c), universe_fn,
                                 full_start, dates[-1], window=WINDOW, step=STEP)
    full_losses[c] = {f.date: f.realized_var for f in r.folds}

val_c = {}
for year in range(EVAL_START_YEAR, 2027):
    cutoff = pd.Timestamp(f"{year}-01-01")
    val_c[year] = min(
        GRID,
        key=lambda c: np.mean([v for d, v in full_losses[c].items() if d < cutoff]),
    )
print("validation-selected c per year:", val_c)

def valselect_fn(R, symbols, as_of):
    return backtest.covariance_weight_fn(
        lambda R_: baselines.gerber_cov(R_, val_c[as_of.year], NORM)
    )(R, symbols, as_of)

runs, rows = {}, []
for name, wfn in ([("learned", learned_fn), ("val_select", valselect_fn)]
                  + [(f"c={c}", fixed_fn(c)) for c in GRID]):
    t0 = time.time()
    runs[name] = backtest.run_walkforward(
        name, rets, wfn, universe_fn, eval_start, dates[-1],
        window=WINDOW, step=STEP)
    m = runs[name].metrics(cost_bps=10)
    rows.append({"threshold": name, "ann_vol_net": round(m["ann_vol_net"], 5),
                 "sharpe_net": round(m["sharpe_net"], 3),
                 "mdd": round(m["max_drawdown"], 4),
                 "turnover": round(m["avg_turnover"], 3)})
    print(f"  {name:10s} vol_net={m['ann_vol_net']*100:6.2f}%  "
          f"sharpe={m['sharpe_net']:+.2f}  to={m['avg_turnover']:.2f} "
          f"({time.time() - t0:.0f}s)")

print("\n== DM vs each fixed threshold (per-fold realized variance) ==")
dm_rows = []
for method in ["learned", "val_select"]:
    for c in GRID:
        dm, p = stats.diebold_mariano(runs[method].fold_losses(),
                                      runs[f"c={c}"].fold_losses())
        dm_rows.append({"method": method, "vs_c": c,
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})
        print(f"  {method:10s} vs c={c}: DM={dm:+.2f} p={p:.4f} "
              f"({method + ' better' if dm < 0 else 'fixed better'})")
dm, p = stats.diebold_mariano(runs["learned"].fold_losses(),
                              runs["val_select"].fold_losses())
print(f"  learned vs val_select: DM={dm:+.2f} p={p:.4f}")

pd.DataFrame(rows).to_csv(RESULTS / f"threshold_grid_{TAG}.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / f"threshold_grid_dm_{TAG}.csv", index=False)
print(f"\nwritten to {RESULTS}")
