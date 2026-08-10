"""Train/eval ridge consistency check.

Training used an ABSOLUTE ridge (Sigma + 1e-4 I ≈ 77% of the mean diagonal
for daily returns) while evaluation uses a SCALE-RELATIVE ridge
(Sigma + 1e-4 * tr(Sigma)/N * I ≈ 0.01%). Thresholds were therefore learned
under a far more regularized decision layer than the one they are deployed in.

This retrains T-global at both universes with a scale-matched training ridge
and compares learned thresholds and OOS performance against the original run.
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
from diffgerber.soft_gerber import SoftGerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
WINDOW, STEP, EPOCHS = 252, 21, 40
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def gmv_scaled(Sigma, ridge=1e-4):
    """Scale-relative ridge — matches backtest.gmv_from_cov exactly."""
    N = Sigma.shape[-1]
    scale = torch.diagonal(Sigma, dim1=-2, dim2=-1).mean(-1)
    eye = torch.eye(N, dtype=Sigma.dtype, device=Sigma.device)
    ones = torch.ones(Sigma.shape[:-1] + (1,), dtype=Sigma.dtype, device=Sigma.device)
    x = torch.linalg.solve(Sigma + ridge * scale * eye, ones).squeeze(-1)
    return x / x.sum(-1, keepdim=True)


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

    module = GlobalThreshold(0.5)
    layer = SoftGerber("gs2022", straight_through=True)
    opt = torch.optim.Adam(module.parameters(), lr=0.02)
    learned_new = {}
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
                w = gmv_scaled(Sig, ridge=1e-4)          # <-- matched ridge
                port = f["Rf"] @ w
                loss = loss + (port ** 2).mean()
            (loss / len(train)).backward()
            opt.step()
        learned_new[year] = float(module().detach())
        print(f"  top{topn} matched-ridge {year}: c={learned_new[year]:.3f} "
              f"({len(train)} folds, {time.time()-t0:.0f}s)", flush=True)

    tag = "gs2022" if topn == 100 else "gs2022_top400"
    snap = json.loads((RESULTS / f"scalar_thresholds_{tag}.json").read_text())
    learned_old = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
                   for k in snap if k.startswith("tglobal_")}

    def gfn(cmap):
        def fn(R, s_, d_):
            c = cmap[d_.year] if isinstance(cmap, dict) else cmap
            return backtest.covariance_weight_fn(
                lambda R_: baselines.gerber_cov(R_, c, "gs2022"))(R, s_, d_)
        return fn

    runs = {}
    for name, cmap in [("matched_ridge", learned_new),
                       ("original", learned_old),
                       ("fixed_0.5", 0.5)]:
        runs[name] = backtest.run_walkforward(name, rets, gfn(cmap), uni, eval_start,
                                              dates[-1], window=WINDOW, step=STEP)
        m = runs[name].metrics(cost_bps=10)
        rows.append({"universe": f"top{topn}", "training": name,
                     "ann_vol_net": round(m["ann_vol_net"], 5),
                     "sharpe_net": round(m["sharpe_net"], 3),
                     "turnover": round(m["avg_turnover"], 3)})
        print(f"  top{topn} {name:14s} vol={m['ann_vol_net']*100:6.2f}%  "
              f"sharpe={m['sharpe_net']:+.2f}", flush=True)
    dm, p = stats.diebold_mariano(runs["matched_ridge"].fold_losses(),
                                  runs["original"].fold_losses())
    print(f"  top{topn} matched vs original: DM={dm:+.2f} p={p:.4f}")
    print(f"  top{topn} c  original: " +
          str({y: round(c, 3) for y, c in sorted(learned_old.items())}))
    print(f"  top{topn} c  matched : " +
          str({y: round(c, 3) for y, c in sorted(learned_new.items())}), flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "ridge_consistency.csv", index=False)
print("done.")
