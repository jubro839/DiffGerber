"""E1: analytical nonlinear shrinkage (ANS) baseline at both universes."""

import json
import math
import pathlib
import sys

import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats

torch.set_default_dtype(torch.float64)
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return("2015-01-01", "2026-07-28", symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index
reb = [dates[i] for i in range(252, len(dates) - 1, 21)]
eval_start = min(d for d in reb if d.year >= 2017)

rows = []
for topn, tag in [(100, "gs2022"), (400, "gs2022_top400")]:
    _m = {}
    def uni(d, topn=topn, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe(d, topn, px, mem)
        return _m[d]
    snap = json.loads((RESULTS / f"scalar_thresholds_{tag}.json").read_text())
    cg = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
          for k in snap if k.startswith("tglobal_")}

    contenders = {
        "ans": backtest.covariance_weight_fn(baselines.analytical_nonlinear_shrinkage),
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "tglobal": lambda R, s_, d_: backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, cg[d_.year], "gs2022"))(R, s_, d_),
    }
    runs = {}
    for name, wfn in contenders.items():
        runs[name] = backtest.run_walkforward(name, rets, wfn, uni, eval_start,
                                              dates[-1], window=252, step=21)
        m = runs[name].metrics(cost_bps=10)
        rows.append({"universe": f"top{topn}", "method": name,
                     "ann_vol_net": round(m["ann_vol_net"], 5),
                     "sharpe_net": round(m["sharpe_net"], 3),
                     "mdd": round(m["max_drawdown"], 4),
                     "turnover": round(m["avg_turnover"], 3)})
        print(f"top{topn} {name:12s} vol={m['ann_vol_net']*100:6.2f}%  "
              f"sharpe={m['sharpe_net']:+.2f}  to={m['avg_turnover']:.2f}")
    for a, b in [("tglobal", "ans"), ("gerber_c0.5", "ans")]:
        dm, p = stats.diebold_mariano(runs[a].fold_losses(), runs[b].fold_losses())
        print(f"  top{topn} {a} vs {b}: DM={dm:+.2f} p={p:.4f} "
              f"({a + ' better' if dm < 0 else b + ' better'})")

pd.DataFrame(rows).to_csv(RESULTS / "e1_ans.csv", index=False)
print("E1 done.")
