"""Sector-balanced universes at N in {30, 50, 100}.

Universe: point-in-time S&P members, slots allocated equally across sectors,
filled by market cap within sector (data.pit_universe_sector). Protocol is the
frozen one: window 252, step 21, eval 2017+, 10bps, scale-relative ridge 1e-3,
straight-through training, fresh Adam per refit, lr 0.02, epochs 40,
horizon-aware refit boundary.

Metrics reported: Ann Return, Ann Vol, Sharpe, Sortino, MDD (+ turnover).
Arms: fixed Gerber c=0.5, Ledoit-Wolf, ANS, HRP, T-global, T-asym.
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
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import AsymThreshold, GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [30, 50, 100]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index

METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]

all_rows, dm_rows, c_rows = [], [], []
for N in SIZES:
    print(f"===== N={N} (sector-balanced) =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        })
    in_range = [f["date"] for f in folds if 2017 <= f["date"].year <= LAST_YEAR]
    eval_start, eval_end = min(in_range), max(in_range) + pd.Timedelta(days=1)

    # ---- train T-global and T-asym (frozen protocol) ----
    learned = {}
    for kind, make in [("tglobal", lambda: GlobalThreshold(0.5)),
                       ("tasym", lambda: AsymThreshold(0.5))]:
        torch.manual_seed(0)
        module = make()
        layer = SoftGerber("gs2022", straight_through=True)
        per_year = {}
        for year in range(2017, LAST_YEAR + 1):
            opt = torch.optim.Adam(module.parameters(), lr=0.02)
            cutoff = pd.Timestamp(f"{year}-01-01")
            train = [f for f in folds
                     if dates[min(dates.get_loc(f["date"]) + STEP,
                                  len(dates) - 1)] < cutoff]
            t0 = time.time()
            for ep in range(EPOCHS):
                tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
                opt.zero_grad()
                loss = 0.0
                for f in train:
                    s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                    if kind == "tglobal":
                        G = layer(f["Rw"] / s, module(), tau)
                    else:
                        cu, cd = module()
                        G = layer(f["Rw"] / s, cu, tau, c_down=cd)
                    Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                    loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
                (loss / len(train)).backward()
                opt.step()
            with torch.no_grad():
                if kind == "tglobal":
                    per_year[year] = (float(module()), None)
                else:
                    cu, cd = module()
                    per_year[year] = (float(cu), float(cd))
            print(f"  {kind} {year}: c={per_year[year]} "
                  f"({len(train)} folds, {time.time()-t0:.0f}s)", flush=True)
        learned[kind] = per_year
        for y, (cu, cd) in per_year.items():
            c_rows.append({"N": N, "module": kind, "year": y,
                           "c_up": round(cu, 4),
                           "c_down": None if cd is None else round(cd, 4)})

    def learned_fn(kind):
        def fn(R, symbols, as_of):
            cu, cd = learned[kind][as_of.year]
            Rt = torch.as_tensor(R)
            s = Rt.std(0, unbiased=True).clamp_min(1e-8)
            G = hard_gerber(Rt / s, cu, "gs2022", c_down=cd)
            return backtest.gmv_from_cov(
                (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
        return fn

    contenders = {
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
        "tglobal": learned_fn("tglobal"),
        "tasym": learned_fn("tasym"),
    }
    runs = {}
    for name, wfn in contenders.items():
        runs[name] = backtest.run_walkforward(name, rets, wfn, uni, eval_start,
                                              eval_end, window=WINDOW, step=STEP)
        m = runs[name].metrics(cost_bps=10)
        all_rows.append({"N": N, "method": name,
                         **{k: round(float(m[k]), 5) for k in METRIC_KEYS}})
        print(f"  N={N} {name:12s} ret={m['ann_ret_net']*100:6.2f}%  "
              f"vol={m['ann_vol_net']*100:6.2f}%  SR={m['sharpe_net']:+.2f}  "
              f"So={m['sortino_net']:+.2f}  mdd={m['max_drawdown']*100:6.1f}%",
              flush=True)
    for a in ["tglobal", "tasym"]:
        for b in ["gerber_c0.5", "ans", "ledoit_wolf"]:
            dm, p = stats.diebold_mariano(runs[a].fold_losses(),
                                          runs[b].fold_losses())
            dm_rows.append({"N": N, "method": a, "vs": b,
                            "dm": round(float(dm), 3), "p": round(float(p), 5)})
            print(f"    {a} vs {b}: DM={dm:+.2f} p={p:.4f}", flush=True)

pd.DataFrame(all_rows).to_csv(RESULTS / "sector_universe_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "sector_universe_dm.csv", index=False)
pd.DataFrame(c_rows).to_csv(RESULTS / "sector_universe_thresholds.csv", index=False)
print("\nsector-universe experiment done.")
