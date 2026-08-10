"""Is there a threshold that makes the HRP decision beat HRP and 1/N?

DG-HRP currently reuses the threshold trained against the minimum-variance
objective, which contradicts the paper's own thesis: if the threshold is a
decision parameter, the hierarchical allocator deserves its own. Before
building that, measure whether the headroom exists at all.

For each universe we sweep fixed thresholds inside the HRP layer and compare
the best one against plain HRP (sample correlation) and equal weight. If even
the ex-post best threshold cannot beat those two, no learned threshold will,
and the direction is dead regardless of how it is trained.
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data
from diffgerber.soft_gerber import hard_gerber

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,70,100").split(",")]
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
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

rows = []
for N in SIZES:
    print(f"===== sector N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    def hrp_gerber(c):
        def fn(R, symbols, as_of):
            Rt = torch.as_tensor(R)
            s = Rt.std(0, unbiased=True).clamp_min(1e-8)
            G = hard_gerber(Rt / s, c, "gs2022").numpy()
            return np.asarray(baselines.hrp_weights(R, corr=G))
        return fn

    variants = {f"hrp_gerber_c{c}": hrp_gerber(c) for c in GRID}
    variants["hrp_sample"] = lambda R, s, d: baselines.hrp_weights(R)
    variants["equal_weight"] = lambda R, s, d: np.full(R.shape[1], 1.0 / R.shape[1])

    for name, fn in variants.items():
        r = backtest.run_walkforward(name, rets, fn, uni, report_start, run_end,
                                     window=WINDOW, step=STEP)
        m = r.metrics(cost_bps=10.0)
        rows.append({"N": N, "rule": name,
                     "ann_ret_net": round(float(m["ann_ret_net"]), 5),
                     "ann_vol_net": round(float(m["ann_vol_net"]), 5),
                     "sharpe_net": round(float(m["sharpe_net"]), 5),
                     "sortino_net": round(float(m["sortino_net"]), 5),
                     "max_drawdown": round(float(m["max_drawdown"]), 5)})
        print(f"  {name:20s} ret={m['ann_ret_net']*100:6.2f}%  "
              f"vol={m['ann_vol_net']*100:6.2f}%  SR={m['sharpe_net']:+.3f}",
              flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "hrp_threshold_ceiling.csv", index=False)

print("\n=== can ANY threshold in the HRP layer beat HRP and 1/N? ===")
for N in SIZES:
    s = df[df.N == N].set_index("rule")
    g = s[s.index.str.startswith("hrp_gerber_")]
    best = g.loc[g.sharpe_net.idxmax()]
    hrp, ew = s.loc["hrp_sample"], s.loc["equal_weight"]
    print(f"N={N}: oracle-c {best.name.replace('hrp_gerber_c','c=')} "
          f"SR={best.sharpe_net:.3f} | HRP {hrp.sharpe_net:.3f} "
          f"| 1/N {ew.sharpe_net:.3f}  -> "
          f"vs HRP {best.sharpe_net-hrp.sharpe_net:+.3f}, "
          f"vs 1/N {best.sharpe_net-ew.sharpe_net:+.3f}")
print("\nHRP threshold ceiling done.")
