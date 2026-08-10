"""E4: decision-dependent thresholds — does the optimal co-movement
definition depend on the downstream investment decision?

Train T-global under a LONG-ONLY GMV layer (cvxpylayers QP) at top-100 and
compare the learned threshold path against the long-short-learned path
(scalar_thresholds_gs2022.json). Then evaluate three policies under
long-only: learned-for-long-only c, fixed 0.5, and the long-short c.
"""

import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import LongOnlyGMV, decision_loss
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
WINDOW, STEP, TOP_N, EPOCHS = 252, 21, 100, 40
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
        "date": dates[i], "syms": syms,
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
    })

lo_layer = LongOnlyGMV(TOP_N)

def longonly_weights_soft(Rw, c, tau, layer):
    s = Rw.std(0, unbiased=True).clamp_min(1e-8)
    G = layer(Rw / s, c, tau)
    Sigma = s.unsqueeze(-1) * G * s.unsqueeze(-2)
    return lo_layer(Sigma)

# ---- train under long-only decision layer ----
module = GlobalThreshold(0.5)
soft = SoftGerber("gs2022", straight_through=True)
learned_lo = {}
for year in range(2017, 2027):
    opt = torch.optim.Adam(module.parameters(), lr=0.02)
    cutoff = pd.Timestamp(f"{year}-01-01")
    train = [f for f in folds
             if dates[min(dates.get_loc(f["date"]) + STEP, len(dates) - 1)] < cutoff]
    t0 = time.time()
    for ep in range(EPOCHS):
        tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
        opt.zero_grad()
        loss = 0.0
        for f in train:
            w = longonly_weights_soft(f["Rw"], module(), tau, soft)
            loss = loss + decision_loss(w, f["Rf"])
        (loss / len(train)).backward()
        opt.step()
    learned_lo[year] = float(module().detach())
    print(f"  long-only {year}: c={learned_lo[year]:.3f} "
          f"({len(train)} folds, {time.time()-t0:.0f}s)", flush=True)

ls = json.loads((RESULTS / "scalar_thresholds_gs2022.json").read_text())
learned_ls = {int(k.split("_")[1]): math.log(1 + math.exp(ls[k]["raw"]))
              for k in ls if k.startswith("tglobal_")}
print("long-short c:", {y: round(c, 3) for y, c in sorted(learned_ls.items())})
print("long-only  c:", {y: round(c, 3) for y, c in sorted(learned_lo.items())})

# ---- evaluate three policies under long-only GMV ----
def lo_weight_fn(c_map_or_val):
    def fn(R, symbols, as_of):
        c = c_map_or_val[as_of.year] if isinstance(c_map_or_val, dict) else c_map_or_val
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        Sigma = s.unsqueeze(-1) * G * s.unsqueeze(-2)
        with torch.no_grad():
            w = lo_layer(Sigma)
        return w.numpy()
    return fn

eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)
rows, runs = [], {}
for name, wfn in [("lo_learned", lo_weight_fn(learned_lo)),
                  ("lo_fixed0.5", lo_weight_fn(0.5)),
                  ("lo_from_longshort", lo_weight_fn(learned_ls))]:
    runs[name] = backtest.run_walkforward(name, rets, wfn, uni, eval_start,
                                          dates[-1], window=WINDOW, step=STEP)
    m = runs[name].metrics(cost_bps=10)
    rows.append({"method": name, "ann_vol_net": round(m["ann_vol_net"], 5),
                 "sharpe_net": round(m["sharpe_net"], 3),
                 "mdd": round(m["max_drawdown"], 4),
                 "turnover": round(m["avg_turnover"], 3)})
    print(f"  {name:18s} vol={m['ann_vol_net']*100:6.2f}%  sharpe={m['sharpe_net']:+.2f}  "
          f"to={m['avg_turnover']:.2f}", flush=True)

# The manuscript quotes this p-value for the "learning is harmful under a
# long-only layer" boundary, so it has to leave an artifact, not just stdout.
dm_rows = []
for base in ["lo_fixed0.5", "lo_from_longshort"]:
    dm, p = stats.diebold_mariano(runs["lo_learned"].fold_losses(),
                                  runs[base].fold_losses())
    dm_rows.append({"method": "lo_learned", "vs": base,
                    "dm": round(float(dm), 3), "p": round(float(p), 5)})
    print(f"  lo_learned vs {base}: DM={dm:+.2f} p={p:.4f}")
pd.DataFrame(dm_rows).to_csv(RESULTS / "decision_dependence_dm.csv", index=False)

pd.DataFrame(rows).to_csv(RESULTS / "decision_dependence.csv", index=False)
pd.DataFrame({"year": list(learned_lo), "c_longonly": list(learned_lo.values()),
              "c_longshort": [learned_ls[y] for y in learned_lo]}
             ).to_csv(RESULTS / "decision_dependence_thresholds.csv", index=False)
print("E4 done.")
