"""E5: threshold non-transferability — apply thresholds learned on one
universe to the other. If cross-applied thresholds underperform in-situ
learned ones, thresholds are universe-specific and must be learned in place.
"""

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

def load_c(tag):
    snap = json.loads((RESULTS / f"scalar_thresholds_{tag}.json").read_text())
    return {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
            for k in snap if k.startswith("tglobal_")}

C = {"top100": load_c("gs2022"), "top400": load_c("gs2022_top400")}

rows, runs = [], {}
for eval_topn in [100, 400]:
    _m = {}
    def uni(d, topn=eval_topn, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe(d, topn, px, mem)
        return _m[d]
    for source in ["top100", "top400"]:
        cmap = C[source]
        def wfn(R, s_, d_, cmap=cmap):
            return backtest.covariance_weight_fn(
                lambda R_: baselines.gerber_cov(R_, cmap[d_.year], "gs2022"))(R, s_, d_)
        key = f"eval_top{eval_topn}__c_from_{source}"
        runs[key] = backtest.run_walkforward(key, rets, wfn, uni, eval_start,
                                             dates[-1], window=252, step=21)
        m = runs[key].metrics(cost_bps=10)
        rows.append({"eval_universe": f"top{eval_topn}", "thresholds_from": source,
                     "ann_vol_net": round(m["ann_vol_net"], 5),
                     "sharpe_net": round(m["sharpe_net"], 3)})
        print(f"eval=top{eval_topn:3d}  c from {source}: vol={m['ann_vol_net']*100:6.2f}%  "
              f"sharpe={m['sharpe_net']:+.2f}")

for ev in [100, 400]:
    insitu = runs[f"eval_top{ev}__c_from_top{ev}"].fold_losses()
    other = "top400" if ev == 100 else "top100"
    cross = runs[f"eval_top{ev}__c_from_{other}"].fold_losses()
    dm, p = stats.diebold_mariano(insitu, cross)
    print(f"eval=top{ev}: in-situ vs transferred DM={dm:+.2f} p={p:.4f} "
          f"({'in-situ better' if dm < 0 else 'transferred better'})")

pd.DataFrame(rows).to_csv(RESULTS / "transfer_matrix.csv", index=False)
print("E5 done.")
