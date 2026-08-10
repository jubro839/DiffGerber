"""How much is there to win from the threshold at all?

Before redesigning anything, measure the ceiling. For each universe we sweep a
grid of fixed thresholds and record the ex-post best one -- the value a
clairvoyant would have chosen. Three quantities follow:

    oracle - default   the most any threshold rule could have gained
    learned - default  what our procedure actually captured
    oracle - learned   what is left on the table for a better architecture

If the first number is small, no architecture can produce a large effect on
this data and the honest conclusion is that the headroom, not the method, is
the binding constraint.
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
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,100").split(",")]
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

    def fixed_fn(c):
        def fn(R, symbols, as_of):
            Rt = torch.as_tensor(R)
            s = Rt.std(0, unbiased=True).clamp_min(1e-8)
            G = hard_gerber(Rt / s, c, "gs2022")
            return backtest.gmv_from_cov(
                (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
        return fn

    for c in GRID:
        r = backtest.run_walkforward(f"c{c}", rets, fixed_fn(c), uni,
                                     report_start, run_end,
                                     window=WINDOW, step=STEP)
        m = r.metrics(cost_bps=10.0)
        rows.append({"N": N, "rule": f"fixed_c={c}", "c": c,
                     "ann_ret_net": round(float(m["ann_ret_net"]), 5),
                     "ann_vol_net": round(float(m["ann_vol_net"]), 5),
                     "sharpe_net": round(float(m["sharpe_net"]), 5),
                     "max_drawdown": round(float(m["max_drawdown"]), 5)})
        print(f"  c={c:<5} vol={m['ann_vol_net']*100:6.3f}%  "
              f"SR={m['sharpe_net']:+.3f}", flush=True)

    th = pd.read_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv")
    tg = {int(r.year): float(r.c_up)
          for r in th[th.module == "tglobal"].itertuples()}

    def learned_fn(R, symbols, as_of):
        c = tg.get(as_of.year, 0.5)
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        return backtest.gmv_from_cov(
            (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())

    r = backtest.run_walkforward("learned", rets, learned_fn, uni,
                                 report_start, run_end, window=WINDOW, step=STEP)
    m = r.metrics(cost_bps=10.0)
    rows.append({"N": N, "rule": "learned", "c": np.nan,
                 "ann_ret_net": round(float(m["ann_ret_net"]), 5),
                 "ann_vol_net": round(float(m["ann_vol_net"]), 5),
                 "sharpe_net": round(float(m["sharpe_net"]), 5),
                 "max_drawdown": round(float(m["max_drawdown"]), 5)})
    print(f"  learned  vol={m['ann_vol_net']*100:6.3f}%  "
          f"SR={m['sharpe_net']:+.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "threshold_ceiling.csv", index=False)

print("\n=== headroom (Sharpe and volatility) ===")
for N in SIZES:
    s = df[df.N == N]
    grid = s[s.rule.str.startswith("fixed")]
    default = grid[grid.c == 0.5].iloc[0]
    orc_sr = grid.loc[grid.sharpe_net.idxmax()]
    orc_vol = grid.loc[grid.ann_vol_net.idxmin()]
    learned = s[s.rule == "learned"].iloc[0]
    print(f"N={N}")
    print(f"  Sharpe : default {default.sharpe_net:.3f} | "
          f"learned {learned.sharpe_net:.3f} | "
          f"oracle {orc_sr.sharpe_net:.3f} (c={orc_sr.c})  -> "
          f"max gain {orc_sr.sharpe_net-default.sharpe_net:+.3f}, "
          f"captured {learned.sharpe_net-default.sharpe_net:+.3f}, "
          f"left {orc_sr.sharpe_net-learned.sharpe_net:+.3f}")
    print(f"  Vol(%) : default {default.ann_vol_net*100:.3f} | "
          f"learned {learned.ann_vol_net*100:.3f} | "
          f"oracle {orc_vol.ann_vol_net*100:.3f} (c={orc_vol.c})  -> "
          f"max gain {(default.ann_vol_net-orc_vol.ann_vol_net)*100:+.3f}, "
          f"captured {(default.ann_vol_net-learned.ann_vol_net)*100:+.3f}")
print("\nthreshold ceiling done.")
