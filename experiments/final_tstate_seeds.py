"""Final round B: T-state v1 multi-seed at top-400 (headline integrity).

Retrains the 3-feature/hidden-16 T-state at top-400 with seeds {0,1,2}
(NOCLIP training as in the headline run), saves per-seed metrics, and
persists seed-0 fold losses + snapshots as the canonical v1 record.
"""

import copy
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, data
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import StateThresholdNet

torch.set_default_dtype(torch.float64)
WINDOW, STEP, TOP_N, EPOCHS = 252, 21, 400, 40
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
        "date": dates[i], "i": i,
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        "Fw": torch.as_tensor(feats_full.iloc[i - WINDOW : i].values),
    })
eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)

rows = []
for seed in [0, 1, 2]:
    torch.manual_seed(seed)
    module = StateThresholdNet(3, None, hidden=16)
    layer = SoftGerber("gs2022", straight_through=True)   # NOCLIP as headline run
    opt = torch.optim.Adam(module.parameters(), lr=3e-3)
    snaps = {}
    for year in range(2017, 2027):
        train = [f for f in folds if f["date"] < pd.Timestamp(f"{year}-01-01")]
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in train:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, module(f["Fw"]), tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(train)).backward()
            opt.step()
        snaps[year] = copy.deepcopy(module.state_dict())
        print(f"  seed{seed} {year} trained ({len(train)} folds, {time.time()-t0:.0f}s)",
              flush=True)

    def wfn(R, symbols, as_of):
        module.load_state_dict(snaps[as_of.year])
        i = dates.get_loc(as_of)
        with torch.no_grad():
            c = module(torch.as_tensor(feats_full.iloc[i - WINDOW : i].values))
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        return backtest.gmv_from_cov((s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())

    r = backtest.run_walkforward(f"tstate_s{seed}", rets, wfn, uni, eval_start,
                                 dates[-1], window=WINDOW, step=STEP)
    m = r.metrics(cost_bps=10)
    rows.append({"seed": seed, "ann_vol_net": round(m["ann_vol_net"], 5),
                 "sharpe_net": round(m["sharpe_net"], 3),
                 "mdd": round(m["max_drawdown"], 4)})
    print(f"seed {seed}: vol={m['ann_vol_net']*100:.2f}% sharpe={m['sharpe_net']:+.2f}",
          flush=True)
    if seed == 0:
        torch.save(snaps, RESULTS / "snapshots_tstate_v1_top400.pt")
        pd.DataFrame([{"method": "diffgerber_tstate_v1", "date": str(f.date.date()),
                       "loss": f.realized_var} for f in r.folds]
                     ).to_csv(RESULTS / "fold_losses_tstate_v1_top400.csv", index=False)

pd.DataFrame(rows).to_csv(RESULTS / "tstate_seeds_top400.csv", index=False)
print("B done.")
