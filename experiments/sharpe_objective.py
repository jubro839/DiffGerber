"""Sharpe-objective ablation: train T-global to maximize realized Sharpe
(instead of minimizing realized variance) and test OOS generalization.

Loss per refit: -Sharpe of the concatenated daily portfolio returns across
all training folds. Same protocol otherwise (annual expanding refit,
warm start, eval 2017+, GS2022).
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
WINDOW, STEP, EPOCHS = 252, 21, 40
TOP_N = int(os.environ.get("DG_TOPN", 100))
NOCLIP = TOP_N >= 400
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

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

folds = []
for i in range(WINDOW, len(dates) - 1, STEP):
    syms = uni(dates[i])
    folds.append({
        "date": dates[i],
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
    })

module = GlobalThreshold(0.5)
layer = SoftGerber("gs2022", straight_through=True)
learned_c = {}
for year in range(2017, 2027):
    opt = torch.optim.Adam(module.parameters(), lr=0.02)
    cutoff = pd.Timestamp(f"{year}-01-01")
    train = [f for f in folds
             if dates[min(dates.get_loc(f["date"]) + STEP, len(dates) - 1)] < cutoff]
    t0 = time.time()
    for ep in range(EPOCHS):
        tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
        opt.zero_grad()
        ports = []
        for f in train:
            s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
            G = layer(f["Rw"] / s, module(), tau)
            Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
            w = gmv_closed_form(Sig)
            ports.append(f["Rf"] @ w)
        port = torch.cat(ports)
        sharpe = port.mean() / port.std(unbiased=True).clamp_min(1e-12) * np.sqrt(252)
        (-sharpe).backward()
        opt.step()
    learned_c[year] = float(module().detach())
    print(f"  top{TOP_N} sharpe-obj {year}: c={learned_c[year]:.3f} "
          f"({len(train)} folds, {time.time()-t0:.0f}s)", flush=True)

eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)

def gerber_fn(c_map_or_val):
    def fn(R, s_, d_):
        c = c_map_or_val[d_.year] if isinstance(c_map_or_val, dict) else c_map_or_val
        return backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, c, "gs2022"))(R, s_, d_)
    return fn

import json, math
snap_tag = "gs2022" if TOP_N == 100 else f"gs2022_top{TOP_N}"
snap = json.loads((RESULTS / f"scalar_thresholds_{snap_tag}.json").read_text())
var_c = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
         for k in snap if k.startswith("tglobal_")}

rows = []
for name, wfn in [("sharpe_trained", gerber_fn(learned_c)),
                  ("variance_trained", gerber_fn(var_c)),
                  ("fixed_0.5", gerber_fn(0.5))]:
    r = backtest.run_walkforward(name, rets, wfn, uni, eval_start, dates[-1],
                                 window=WINDOW, step=STEP)
    m = r.metrics(cost_bps=10)
    rows.append({"universe": f"top{TOP_N}", "method": name,
                 "ann_vol_net": round(m["ann_vol_net"], 5),
                 "sharpe_net": round(m["sharpe_net"], 3),
                 "mdd": round(m["max_drawdown"], 4)})
    print(f"  top{TOP_N} {name:17s} vol={m['ann_vol_net']*100:6.2f}%  "
          f"sharpe={m['sharpe_net']:+.2f}  mdd={m['max_drawdown']*100:6.1f}%", flush=True)

out = RESULTS / f"sharpe_objective_top{TOP_N}.csv"
pd.DataFrame(rows).to_csv(out, index=False)

# The threshold path is the point of this ablation, so persist it rather than
# leaving it on stdout: the manuscript quotes the value it collapses to.
pd.DataFrame([{"universe": f"top{TOP_N}", "year": y,
               "c_sharpe_trained": round(c, 5),
               "c_variance_trained": round(var_c[y], 5)}
              for y, c in sorted(learned_c.items())]).to_csv(
    RESULTS / f"sharpe_objective_thresholds_top{TOP_N}.csv", index=False)
print("sharpe-trained c:", {y: round(c, 3) for y, c in learned_c.items()})
print("done.")
