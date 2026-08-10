"""The constant-threshold ceiling for the sector-balanced universes.

The mixed design measured what a constant threshold can achieve before
claiming anything for learning (the 4x4 grid in mixed_group_experiment.py).
The sector design, which the paper reports as primary, never had that
ceiling: threshold_grid.py measures it only for the market-cap top-100
universe. The discipline was therefore applied asymmetrically, and this
closes the gap.

For each sector-balanced universe (N in {30, 70}) it walks a grid of
constant thresholds through the identical protocol used by every other arm,
so the learned coefficient can be read against the best a fixed choice
could have made with full hindsight -- a restricted reference, not an
adaptive bound, exactly as in Table 2.

Protocol is frozen as everywhere else: window 252 / step 21, evaluation
2017+, 10bp costs, gs2022, scale-relative ridge 1e-3. No training happens
here; every cell is a constant threshold, so there is no seed to set.

Outputs: sector_ceiling_metrics.csv, sector_ceiling_summary.csv
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
SIZES = [30, 70]
# spans the published rule, the region the sector learner selects, and well
# past it, at the same spacing as the top-100 grid in threshold_grid.py
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                             symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index
fold_dates = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)]
report_start = min(d for d in fold_dates if d.year >= 2017)
run_end = max(fold_dates) + pd.Timedelta(days=1)

# what the learner actually chose, to read the ceiling against
_th = pd.read_csv(RESULTS / "headline_thresholds.csv")
LEARNED = {(int(r.N), int(r.year)): float(r.c_up)
           for r in _th[_th.arm == "DG-GMV"].itertuples()}

rows = []
for N in SIZES:
    print(f"===== sector N={N} =====", flush=True)
    _m = {}

    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = list(data.pit_universe_sector(d, N, px, mem))
        return _m[d]

    for c in GRID:
        wfn = backtest.covariance_weight_fn(
            lambda R, c=c: baselines.gerber_cov(R, c, "gs2022"))
        r = backtest.run_walkforward(f"c={c}", rets, wfn, uni,
                                     report_start, run_end,
                                     window=WINDOW, step=STEP)
        m = r.metrics(cost_bps=10)
        rows.append({"N": N, "c": c,
                     "ann_ret_net": round(float(m["ann_ret_net"]), 5),
                     "ann_vol_net": round(float(m["ann_vol_net"]), 5),
                     "sharpe_net": round(float(m["sharpe_net"]), 5),
                     "sortino_net": round(float(m["sortino_net"]), 5),
                     "max_drawdown": round(float(m["max_drawdown"]), 5)})
        print(f"  c={c:<5} vol={m['ann_vol_net']*100:6.3f}%  "
              f"SR={m['sharpe_net']:+.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "sector_ceiling_metrics.csv", index=False)

hm = pd.read_csv(RESULTS / "headline_metrics.csv")
hm = hm[hm.cost_bps == 10]
summary = []
print("\n=== ceiling: what a constant threshold could have done ===")
for N in SIZES:
    s = df[df.N == N].set_index("c")
    best_c = s.ann_vol_net.idxmin()
    learned = hm[(hm.N == N) & (hm.method == "DG-GMV")].iloc[0]
    default = hm[(hm.N == N) & (hm.method == "gerber_c0.5")].iloc[0]
    lc = [v for (n_, y), v in LEARNED.items() if n_ == N and y >= 2019]
    summary.append({"N": N, "best_constant_c": best_c,
                    "best_constant_vol": round(float(s.ann_vol_net.min()) * 100, 4),
                    "learned_vol": round(float(learned.ann_vol_net) * 100, 4),
                    "default_vol": round(float(default.ann_vol_net) * 100, 4),
                    "learned_c_min_2019plus": round(min(lc), 4) if lc else None,
                    "learned_c_max_2019plus": round(max(lc), 4) if lc else None})
    print(f"N={N}: default(c=0.5) {default.ann_vol_net*100:.3f} | "
          f"learned {learned.ann_vol_net*100:.3f} | "
          f"grid-best {s.ann_vol_net.min()*100:.3f} at c={best_c}")
    if lc:
        print(f"       learned c from 2019 spans {min(lc):.2f}-{max(lc):.2f}")
pd.DataFrame(summary).to_csv(RESULTS / "sector_ceiling_summary.csv", index=False)
print("\nsector ceiling done.")
