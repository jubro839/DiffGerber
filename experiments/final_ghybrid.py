"""Final round A: G-hybrid — diagonal-normalized GS2022.

PSD by construction (no clip, no eigh anywhere), denominator approximates
GS2022's pairwise one. If it matches GS2022+clip empirically, the
theory-practice tension dissolves and we gain a new PSD workhorse estimator.
"""

import json
import math
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
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
WINDOW, STEP, EPOCHS = 252, 21, 40
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

# PSD sanity for the soft hybrid
worst = 1.0
layer = SoftGerber("gs2019")
for s_ in range(20):
    z = torch.randn(120, 15, generator=torch.Generator().manual_seed(s_))
    worst = min(worst, float(torch.linalg.eigvalsh(layer(z, 0.5, 0.1)).min()))
print(f"soft gs2019 min eig over 20 draws: {worst:.3e} (must be >= -1e-10)")
assert worst > -1e-10

mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return("2015-01-01", "2026-07-28", symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index

rows = []
for topn in [100, 400]:
    _m = {}
    def uni(d, topn=topn, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe(d, topn, px, mem)
        return _m[d]
    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        })
    eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)

    # train T-global under gs2019 (PSD by construction -> no clip, no eigh)
    module = GlobalThreshold(0.5)
    opt = torch.optim.Adam(module.parameters(), lr=0.02)
    learned_h = {}
    for year in range(2017, 2027):
        train = [f for f in folds if f["date"] < pd.Timestamp(f"{year}-01-01")]
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in train:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, module(), tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(train)).backward()
            opt.step()
        learned_h[year] = float(module().detach())
    print(f"top{topn} hybrid learned c:", {y: round(c, 3) for y, c in learned_h.items()})

    tag = "gs2022" if topn == 100 else "gs2022_top400"
    snap = json.loads((RESULTS / f"scalar_thresholds_{tag}.json").read_text())
    gs_c = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
            for k in snap if k.startswith("tglobal_")}

    def gfn(norm, c_map_or_val):
        def fn(R, s_, d_):
            c = c_map_or_val[d_.year] if isinstance(c_map_or_val, dict) else c_map_or_val
            return backtest.covariance_weight_fn(
                lambda R_: baselines.gerber_cov(R_, c, norm))(R, s_, d_)
        return fn

    contenders = {
        "hybrid_c0.5": gfn("gs2019", 0.5),
        "hybrid_c1.0": gfn("gs2019", 1.0),
        "hybrid_c1.5": gfn("gs2019", 1.5),
        "hybrid_learned": gfn("gs2019", learned_h),
        "gs2022_c0.5": gfn("gs2022", 0.5),
        "gs2022_learned": gfn("gs2022", gs_c),
    }
    runs = {}
    for name, wfn in contenders.items():
        runs[name] = backtest.run_walkforward(name, rets, wfn, uni, eval_start,
                                              dates[-1], window=WINDOW, step=STEP)
        m = runs[name].metrics(cost_bps=10)
        rows.append({"universe": f"top{topn}", "method": name,
                     "ann_vol_net": round(m["ann_vol_net"], 5),
                     "sharpe_net": round(m["sharpe_net"], 3),
                     "turnover": round(m["avg_turnover"], 3)})
        print(f"  top{topn} {name:16s} vol={m['ann_vol_net']*100:6.2f}%  "
              f"sharpe={m['sharpe_net']:+.2f}", flush=True)
    for a, b in [("hybrid_learned", "gs2022_learned"), ("hybrid_c0.5", "gs2022_c0.5")]:
        dm, p = stats.diebold_mariano(runs[a].fold_losses(), runs[b].fold_losses())
        print(f"  top{topn} {a} vs {b}: DM={dm:+.2f} p={p:.4f}")

pd.DataFrame(rows).to_csv(RESULTS / "ghybrid_results.csv", index=False)
print("A done.")
