"""Ablation ①: ridge sensitivity of the fixed-c blow-ups at top-400.

Question: does a larger ridge rescue the rank-deficient fixed-threshold
matrices, and does the learned threshold still win at every ridge level?
"""

import json
import math
import os
import pathlib
import sys

import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffgerber import backtest, baselines, data

torch.set_default_dtype(torch.float64)

TOP_N = int(os.environ.get("DG_TOPN", 400))
RIDGES = [1e-4, 1e-3, 1e-2]
CS = [0.5, 1.0, 1.25, 2.25]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
TAG = "gs2022" + (f"_top{TOP_N}" if TOP_N != 100 else "")

mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return("2015-01-01", "2026-07-28", symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index
_m = {}
def uni(d):
    if d not in _m:
        _m[d] = data.pit_universe(d, TOP_N, px, mem)
    return _m[d]

snap = json.loads((RESULTS / f"scalar_thresholds_{TAG}.json").read_text())
learned_c = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
             for k in snap if k.startswith("tglobal_")}

reb = [dates[i] for i in range(252, len(dates) - 1, 21)]
eval_start = min(d for d in reb if d.year >= 2017)

rows = []
for ridge in RIDGES:
    def fixed(c):
        return backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, c, "gs2022"), ridge=ridge)
    def learned(R, s_, d_):
        return backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, learned_c[d_.year], "gs2022"),
            ridge=ridge)(R, s_, d_)
    for name, wfn in [("learned", learned)] + [(f"c={c}", fixed(c)) for c in CS]:
        r = backtest.run_walkforward(f"{name}_r{ridge}", rets, wfn, uni,
                                     eval_start, dates[-1], window=252, step=21)
        m = r.metrics(cost_bps=10)
        rows.append({"ridge": ridge, "method": name,
                     "ann_vol_net": round(m["ann_vol_net"], 5),
                     "sharpe_net": round(m["sharpe_net"], 3),
                     "turnover": round(m["avg_turnover"], 3)})
        print(f"ridge={ridge:g}  {name:8s} vol={m['ann_vol_net']*100:6.2f}%  "
              f"sharpe={m['sharpe_net']:+.2f}  to={m['avg_turnover']:.2f}")

pd.DataFrame(rows).to_csv(RESULTS / f"ablation_ridge_{TAG}.csv", index=False)
print("saved.")
