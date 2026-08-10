"""Significance of the learned threshold against covariance-estimator peers,
across the sector-balanced universe sweep.

The sweep established the pattern; this establishes whether it is real. For
each N we compare the learned threshold against the three estimators that
compete for the same slot in the same minimum-variance layer -- the published
fixed threshold, linear shrinkage and analytical nonlinear shrinkage -- on
per-fold realized variance, and then pool the fold losses across universes for
a single test with six times the data. Pooling is legitimate here because
every universe shares the same rebalance dates and the same decision layer;
only the asset set differs.

Allocation rules that estimate nothing (1/N, HRP) are not in this comparison
by construction: they are not covariance estimators, and they are reported
elsewhere as context.
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.soft_gerber import hard_gerber

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "20,30,40,50,70,100").split(",")]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index
fold_dates = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)
              if dates[i].year <= LAST_YEAR]
report_start = min(d for d in fold_dates if d.year >= 2017)
run_end = max(fold_dates) + pd.Timedelta(days=1)

PEERS = ["gerber_c0.5", "ledoit_wolf", "ans"]
rows, pooled = [], {b: {"ours": [], "peer": []} for b in PEERS}
vol_rows = []

for N in SIZES:
    print(f"===== N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    th = pd.read_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv")
    tg = {int(r.year): float(r.c_up)
          for r in th[th.module == "tglobal"].itertuples()}

    def learned(R, symbols, as_of):
        c = tg.get(as_of.year, 0.5)
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        return backtest.gmv_from_cov(
            (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())

    fns = {
        "learned": learned,
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
    }
    runs = {}
    for name, fn in fns.items():
        runs[name] = backtest.run_walkforward(name, rets, fn, uni, report_start,
                                              run_end, window=WINDOW, step=STEP)
        m = runs[name].metrics(cost_bps=10.0)
        vol_rows.append({"N": N, "method": name,
                         "ann_vol_net": round(float(m["ann_vol_net"]), 5),
                         "sharpe_net": round(float(m["sharpe_net"]), 5),
                         "max_drawdown": round(float(m["max_drawdown"]), 5)})

    a_l = runs["learned"].fold_losses()
    for b in PEERS:
        b_l = runs[b].fold_losses()
        n_ = min(len(a_l), len(b_l))
        dm, p = stats.diebold_mariano(a_l[:n_], b_l[:n_])
        rows.append({"N": N, "vs": b, "dm": round(float(dm), 3),
                     "p": round(float(p), 5),
                     "vol_ours": round(float(runs["learned"].metrics(cost_bps=10)["ann_vol_net"]) * 100, 3),
                     "vol_peer": round(float(runs[b].metrics(cost_bps=10)["ann_vol_net"]) * 100, 3)})
        pooled[b]["ours"].append(a_l[:n_])
        pooled[b]["peer"].append(b_l[:n_])
        print(f"  vs {b:12s} DM={dm:+.2f} p={p:.4f}", flush=True)

print("\n=== pooled across universes ===")
for b in PEERS:
    a = np.concatenate(pooled[b]["ours"])
    c = np.concatenate(pooled[b]["peer"])
    dm, p = stats.diebold_mariano(a, c)
    lo, med, hi = stats.stationary_bootstrap(a - c, np.mean, n_boot=2000)
    rows.append({"N": "pooled", "vs": b, "dm": round(float(dm), 3),
                 "p": round(float(p), 5),
                 "boot_lo": float(f"{lo:.3e}"), "boot_hi": float(f"{hi:.3e}")})
    print(f"  vs {b:12s} DM={dm:+.2f} p={p:.5f}  "
          f"(n={len(a)} folds; bootstrap CI of mean loss difference "
          f"[{lo:.2e}, {hi:.2e}])", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "sector_sweep_tests.csv", index=False)
pd.DataFrame(vol_rows).to_csv(RESULTS / "sector_sweep_vols.csv", index=False)
print("\nsector sweep tests done.")
