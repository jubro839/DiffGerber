"""E2: crypto universe — asset-class generalization on Gerber's home turf.

Top-20 cryptos by history depth (24/7 daily), window 252 obs, step 21 obs,
annual expanding refit of T-global, eval 2017+. Annualization uses 365.
"""

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
WINDOW, STEP, TOP_N, EPOCHS = 252, 21, 20, 40
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

px = data.crypto_prices("2015-01-01", "2026-07-28")
rets = data.log_returns(px)
dates = rets.index
print(f"crypto panel: {px.shape}, {dates.min().date()}..{dates.max().date()}")

_m = {}
def uni(d):
    if d not in _m:
        _m[d] = data.crypto_universe(d, px, n=TOP_N)
    return _m[d]

folds = []
for i in range(WINDOW, len(dates) - 1, STEP):
    syms = uni(dates[i])
    if len(syms) < 5:
        continue
    folds.append({
        "date": dates[i], "syms": syms,
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
    })
print(f"folds: {len(folds)} ({folds[0]['date'].date()}..{folds[-1]['date'].date()}), "
      f"universe sizes {min(len(f['syms']) for f in folds)}-{max(len(f['syms']) for f in folds)}")

# ---- train T-global, annual expanding refit ----
module = GlobalThreshold(0.5)
layer = SoftGerber("gs2022", straight_through=True)
learned_c = {}
for year in range(2017, 2027):
    opt = torch.optim.Adam(module.parameters(), lr=0.02)
    cutoff = pd.Timestamp(f"{year}-01-01")
    train = [f for f in folds
             if dates[min(dates.get_loc(f["date"]) + STEP, len(dates) - 1)] < cutoff]
    if not train:
        learned_c[year] = 0.5
        continue
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
    learned_c[year] = float(module().detach())
    print(f"  {year}: c={learned_c[year]:.3f} ({len(train)} folds, {time.time()-t0:.0f}s)")

eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)

def gerber_fn(c_map_or_val):
    def fn(R, s_, d_):
        c = c_map_or_val[d_.year] if isinstance(c_map_or_val, dict) else c_map_or_val
        return backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, c, "gs2022"))(R, s_, d_)
    return fn

contenders = {
    "sample": backtest.covariance_weight_fn(baselines.sample_cov),
    "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
    "ans": backtest.covariance_weight_fn(baselines.analytical_nonlinear_shrinkage),
    "gerber_c0.5": gerber_fn(0.5),
    "gerber_c1.0": gerber_fn(1.0),
    "gerber_c1.5": gerber_fn(1.5),
    "tglobal": gerber_fn(learned_c),
}
rows, runs = [], {}
for name, wfn in contenders.items():
    runs[name] = backtest.run_walkforward(name, rets, wfn, uni, eval_start,
                                          dates[-1], window=WINDOW, step=STEP)
    m = runs[name].metrics(cost_bps=10, periods=365)
    rows.append({"method": name, "ann_vol_net": round(m["ann_vol_net"], 5),
                 "sharpe_net": round(m["sharpe_net"], 3),
                 "mdd": round(m["max_drawdown"], 4),
                 "turnover": round(m["avg_turnover"], 3)})
    print(f"  {name:12s} vol={m['ann_vol_net']*100:6.2f}%  sharpe={m['sharpe_net']:+.2f}  "
          f"mdd={m['max_drawdown']*100:6.1f}%  to={m['avg_turnover']:.2f}")

for base in ["gerber_c0.5", "ledoit_wolf", "ans"]:
    dm, p = stats.diebold_mariano(runs["tglobal"].fold_losses(), runs[base].fold_losses())
    print(f"  tglobal vs {base:12s}: DM={dm:+.2f} p={p:.4f} "
          f"({'tglobal better' if dm < 0 else base + ' better'})")

pd.DataFrame(rows).to_csv(RESULTS / "crypto_results.csv", index=False)
pd.Series(learned_c).to_csv(RESULTS / "crypto_learned_c.csv")
print("E2 done. learned c by year:", {y: round(c, 3) for y, c in learned_c.items()})
