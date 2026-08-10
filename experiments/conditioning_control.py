"""Principled fix: consistent decision-layer regularization at train AND eval.

Finding that motivated this: with train/eval ridge matched at the original
(negligible) level, the top-400 learned threshold drifts to c≈1.14 for 2020 —
trained on 2016-2019 folds that contain no degenerate regime — and the single
2020-04-07 fold blows up (138% annualized, 12.3x gross leverage). Threshold
learning therefore does NOT by itself protect against high-dimensional
conditioning failure; the protection in the original run came from an
accidental heavy training ridge.

Here every method (learned and baseline) uses the SAME scale-relative ridge
in training and evaluation, swept over levels, so the decision layer's
conditioning is an explicit, disclosed design parameter.
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
from diffgerber.soft_gerber import SoftGerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP, EPOCHS = 252, 21, 40
TOP_N = int(os.environ.get("DG_TOPN", 400))
RIDGES = [1e-4, 1e-3, 1e-2]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def gmv_t(Sigma, ridge):
    N = Sigma.shape[-1]
    scale = torch.diagonal(Sigma, dim1=-2, dim2=-1).mean(-1)
    eye = torch.eye(N, dtype=Sigma.dtype, device=Sigma.device)
    ones = torch.ones(Sigma.shape[:-1] + (1,), dtype=Sigma.dtype, device=Sigma.device)
    x = torch.linalg.solve(Sigma + ridge * scale * eye, ones).squeeze(-1)
    return x / x.sum(-1, keepdim=True)


mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                             symbols=sorted(mem["fmp_symbol"].unique()))
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
eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)

rows = []
for ridge in RIDGES:
    torch.manual_seed(0)
    module = GlobalThreshold(0.5)
    layer = SoftGerber("gs2022", straight_through=True)
    learned = {}
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
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, module(), tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + ((f["Rf"] @ gmv_t(Sig, ridge)) ** 2).mean()
            (loss / len(train)).backward()
            opt.step()
        learned[year] = float(module().detach())
    print(f"ridge={ridge:g} learned c: "
          f"{ {y: round(c,3) for y,c in learned.items()} }", flush=True)

    def wf(name, cov_fn):
        r = backtest.run_walkforward(
            name, rets,
            lambda R, s_, d_: backtest.gmv_from_cov(cov_fn(R, d_), ridge=ridge),
            uni, eval_start, dates[-1], window=WINDOW, step=STEP)
        m = r.metrics(cost_bps=10)
        maxlev = max(np.abs(f.weights).sum() for f in r.folds)
        worst = max(np.sqrt(f.realized_var * 252) for f in r.folds)
        rows.append({"ridge": ridge, "method": name,
                     "ann_vol_net": round(m["ann_vol_net"], 5),
                     "sharpe_net": round(m["sharpe_net"], 3),
                     "max_gross_lev": round(float(maxlev), 2),
                     "worst_fold_vol": round(float(worst), 4)})
        print(f"  ridge={ridge:g} {name:14s} vol={m['ann_vol_net']*100:6.2f}%  "
              f"sharpe={m['sharpe_net']:+.2f}  maxLev={maxlev:5.2f}  "
              f"worstFold={worst*100:6.1f}%", flush=True)
        return r

    runs = {
        "learned": wf("learned", lambda R, d: baselines.gerber_cov(R, learned[d.year], "gs2022")),
        "fixed_0.5": wf("fixed_0.5", lambda R, d: baselines.gerber_cov(R, 0.5, "gs2022")),
        "fixed_1.25": wf("fixed_1.25", lambda R, d: baselines.gerber_cov(R, 1.25, "gs2022")),
        "ledoit_wolf": wf("ledoit_wolf", lambda R, d: baselines.ledoit_wolf_cov(R)),
        "ans": wf("ans", lambda R, d: baselines.analytical_nonlinear_shrinkage(R)),
    }
    for base in ["fixed_0.5", "ledoit_wolf", "ans"]:
        dm, p = stats.diebold_mariano(runs["learned"].fold_losses(),
                                      runs[base].fold_losses())
        print(f"    learned vs {base:12s}: DM={dm:+.2f} p={p:.4f}", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / f"conditioning_control_top{TOP_N}.csv", index=False)
print("done.")
