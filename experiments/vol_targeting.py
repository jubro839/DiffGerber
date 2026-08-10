"""Final round C: vol-targeting evaluation — translating covariance accuracy
into risk-control quality.

Each method's GMV weights are scaled to a 10% annualized predicted vol
(leverage capped at 3x). Judged on: realized vol vs target, mean per-fold
|realized - target| (risk-control error), and net Sharpe after targeting.
"""

import json
import math
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
TARGET = 0.10
MAX_LEV = 3.0
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return("2015-01-01", "2026-07-28", symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index
reb = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)]
eval_start = min(d for d in reb if d.year >= 2017)

def targeted(cov_fn):
    def fn(R, symbols, as_of):
        Sigma = cov_fn(R, as_of)
        w = backtest.gmv_from_cov(Sigma)
        pred_vol = np.sqrt(max(w @ Sigma @ w, 1e-12) * 252)
        lev = min(TARGET / pred_vol, MAX_LEV)
        return w * lev
    return fn

def gerber_cov_fn(c_map_or_val, norm="gs2022"):
    def fn(R, as_of):
        c = c_map_or_val[as_of.year] if isinstance(c_map_or_val, dict) else c_map_or_val
        return baselines.gerber_cov(R, c, norm)
    return fn

rows = []
for topn in [100, 400]:
    _m = {}
    def uni(d, topn=topn, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe(d, topn, px, mem)
        return _m[d]
    tag = "gs2022" if topn == 100 else "gs2022_top400"
    snap = json.loads((RESULTS / f"scalar_thresholds_{tag}.json").read_text())
    cg = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
          for k in snap if k.startswith("tglobal_")}

    methods = {
        "gerber_c0.5": gerber_cov_fn(0.5),
        "ledoit_wolf": lambda R, a: baselines.ledoit_wolf_cov(R),
        "ans": lambda R, a: baselines.analytical_nonlinear_shrinkage(R),
        "tglobal": gerber_cov_fn(cg),
    }
    for name, cov_fn in methods.items():
        r = backtest.run_walkforward(f"vt_{name}", rets, targeted(cov_fn), uni,
                                     eval_start, dates[-1], window=WINDOW, step=STEP)
        daily = r.daily_returns
        realized = daily.std(ddof=1) * np.sqrt(252)
        fold_vols = [np.sqrt(f.realized_var * 252) for f in r.folds]
        rc_err = float(np.mean([abs(v - TARGET) for v in fold_vols]))
        m = r.metrics(cost_bps=10)
        rows.append({"universe": f"top{topn}", "method": name,
                     "realized_vol": round(float(realized), 5),
                     "risk_ctrl_err": round(rc_err, 5),
                     "sharpe_net": round(m["sharpe_net"], 3),
                     "mdd": round(m["max_drawdown"], 4)})
        print(f"top{topn} {name:12s} realized={realized*100:6.2f}% (target 10%)  "
              f"fold|err|={rc_err*100:5.2f}%  sharpe={m['sharpe_net']:+.2f}  "
              f"mdd={m['max_drawdown']*100:6.1f}%", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "vol_targeting.csv", index=False)
print("C done.")
