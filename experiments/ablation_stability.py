"""Ablation ②: multi-init / multi-seed stability of threshold learning.

- T-global trained from init c in {0.25, 0.5, 1.0}
- T-state trained with torch seeds {0, 1, 2}
Top-100, GS2022, clip on (headline-consistent protocol). Reports per-year
learned thresholds and OOS vol for each run.
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffgerber import backtest, baselines, data
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold, StateThresholdNet

torch.set_default_dtype(torch.float64)

WINDOW, STEP, EPOCHS = 252, 21, 40
TOP_N = int(os.environ.get("DG_TOPN", 100))
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return("2015-01-01", "2026-07-28", symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index

vix = data.index_levels("^VIX", "2015-01-01", "2026-07-28")["close_price"]
vix = vix.reindex(dates).ffill()
csd = rets.std(axis=1)
ewvol = rets.mean(axis=1).ewm(halflife=63).std() * np.sqrt(252)
feats_full = pd.DataFrame({
    "log_vix": np.log(vix / 20.0),
    "log_csd": np.log(csd.clip(lower=1e-4) / 0.015),
    "log_ewvol": np.log(ewvol.clip(lower=1e-3) / 0.15),
}).fillna(0.0)

_m = {}
def uni(d):
    if d not in _m:
        _m[d] = data.pit_universe(d, TOP_N, px, mem)
    return _m[d]

folds = []
for i in range(WINDOW, len(dates) - 1, STEP):
    syms = uni(dates[i])
    folds.append({
        "date": dates[i], "syms": syms, "i": i,
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        "Fw": torch.as_tensor(feats_full.iloc[i - WINDOW : i].values),
    })
eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)


def run_variant(kind, label, make):
    module = make()
    layer = SoftGerber("gs2022", straight_through=True)
    snaps, cs = {}, {}
    for year in range(2017, 2027):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = [f for f in folds
                 if dates[min(dates.get_loc(f["date"]) + STEP, len(dates) - 1)] < cutoff]
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in train:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                c = module() if kind == "tglobal" else module(f["Fw"])
                G = layer(f["Rw"] / s, c, tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(train)).backward()
            opt.step()
        import copy
        snaps[year] = copy.deepcopy(module.state_dict())
        with torch.no_grad():
            c = module() if kind == "tglobal" else module(train[-1]["Fw"])
            cs[year] = float(torch.as_tensor(c).mean())

    def wfn(R, symbols, as_of):
        module.load_state_dict(snaps[as_of.year])
        i = dates.get_loc(as_of)
        with torch.no_grad():
            c = (module() if kind == "tglobal"
                 else module(torch.as_tensor(feats_full.iloc[i - WINDOW : i].values)))
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        return backtest.gmv_from_cov((s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())

    r = backtest.run_walkforward(label, rets, wfn, uni, eval_start, dates[-1],
                                 window=WINDOW, step=STEP)
    m = r.metrics(cost_bps=10)
    print(f"{label:22s} vol={m['ann_vol_net']*100:6.2f}%  sharpe={m['sharpe_net']:+.2f}  "
          f"c(2020)={cs[2020]:.3f} c(2023)={cs[2023]:.3f} c(2026)={cs[2026]:.3f}")
    return {"label": label, "ann_vol_net": round(m["ann_vol_net"], 5),
            "sharpe_net": round(m["sharpe_net"], 3),
            **{f"c_{y}": round(v, 4) for y, v in cs.items()}}


rows = []
for init in [0.25, 0.5, 1.0]:
    torch.manual_seed(0)
    t0 = time.time()
    rows.append(run_variant("tglobal", f"tglobal_init{init}",
                            lambda init=init: GlobalThreshold(init)))
    print(f"  ({time.time()-t0:.0f}s)")
for seed in [0, 1, 2]:
    torch.manual_seed(seed)
    t0 = time.time()
    rows.append(run_variant("tstate", f"tstate_seed{seed}",
                            lambda: StateThresholdNet(3, None, hidden=16)))
    print(f"  ({time.time()-t0:.0f}s)")

pd.DataFrame(rows).to_csv(RESULTS / f"ablation_stability_gs2022_top{TOP_N}.csv", index=False)
print("saved.")
