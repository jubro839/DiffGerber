"""Architecture v2b: two pre-specified rolling selection rules, evaluated once.

The v2 post-mortem showed annual winner-take-all selection with a 3-year
trailing window whipsaws (switches into the winning style only after its run).
Two replacement rules, fixed in advance:

  R1 fast WTA   — at EVERY rebalance, pick the arm with the best Sortino over
                  the trailing 12 folds (~1 year).
  R2 soft blend — at every rebalance, weight the arms in proportion to the
                  positive part of the same trailing Sortino (equal weights if
                  none positive). No tunable constants.

Selection at fold k uses only folds k-12..k-1, whose realization windows end
before fold k's begins (folds tile with no overlap) — no look-ahead.
Arm pool is unchanged from v2 (A1..A5). Static 50/50 A1+A3 is re-reported as
reference. Everything net of 10 bps with turnover computed on the composed
weights, so switching costs are priced natively.
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
from diffgerber.portfolio_layer import (decision_loss, downside_decision_loss,
                                        gmv_closed_form)
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import AsymThreshold, GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,100").split(",")]
COST = 10.0
TRAIL_FOLDS = 12
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index


def train_threshold(folds, kind, loss_fn):
    torch.manual_seed(0)
    module = GlobalThreshold(0.5) if kind == "tglobal" else AsymThreshold(0.5)
    layer = SoftGerber("gs2022", straight_through=True)
    per_year = {2016: (0.5, None if kind == "tglobal" else 0.5)}
    for year in range(2017, LAST_YEAR + 1):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = [f for f in folds
                 if dates[min(dates.get_loc(f["date"]) + STEP,
                              len(dates) - 1)] < cutoff]
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
                loss = loss + loss_fn(gmv_closed_form(Sig), f["Rf"])
            (loss / len(train)).backward()
            opt.step()
        with torch.no_grad():
            if kind == "tglobal":
                per_year[year] = (float(module()), None)
            else:
                cu, cd = module()
                per_year[year] = (float(cu), float(cd))
    return per_year


def gerber_G(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]
rows, sel_rows, dm_rows = [], [], []

for N in SIZES:
    print(f"===== N={N} =====", flush=True)
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
    fold_dates = [f["date"] for f in folds if f["date"].year <= LAST_YEAR]
    run_start = min(fold_dates)
    run_end = max(fold_dates) + pd.Timedelta(days=1)
    report_start = min(d for d in fold_dates if d.year >= 2017)

    t0 = time.time()
    c_var = train_threshold(folds, "tglobal", decision_loss)
    c_asym = train_threshold(folds, "tasym", decision_loss)
    c_dd = train_threshold(folds, "tglobal", downside_decision_loss)
    pd.DataFrame([{"N": N, "module": m_, "year": y, "c_up": v[0], "c_down": v[1]}
                  for m_, cm in [("tglobal", c_var), ("tasym", c_asym),
                                 ("tglobal_dd", c_dd)]
                  for y, v in cm.items()]).to_csv(
        RESULTS / f"arch_v2b_thresholds_N{N}.csv", index=False)
    print(f"  thresholds trained ({time.time()-t0:.0f}s)", flush=True)

    def gmv_fn(cmap):
        def fn(R, symbols, as_of):
            cu, cd = cmap[as_of.year]
            G, s = gerber_G(R, cu, cd)
            return backtest.gmv_from_cov(np.outer(s, s) * G)
        return fn

    def hrp_fn(cmap):
        def fn(R, symbols, as_of):
            cu, cd = cmap[as_of.year]
            G, _ = gerber_G(R, cu, cd)
            return np.asarray(baselines.hrp_weights(R, corr=G))
        return fn

    arm_fns = {
        "A1_gmv_tglobal": gmv_fn(c_var),
        "A2_gmv_tasym": gmv_fn(c_asym),
        "A3_hrp_gerber": hrp_fn(c_var),
        "A4_gmv_dd": gmv_fn(c_dd),
        "A5_hrp_gerber_dd": hrp_fn(c_dd),
    }
    ARMS = list(arm_fns)

    # ---- pass 1: run every arm, collect per-fold net daily series ----
    arm_runs = {}
    arm_fold_net = {}          # arm -> {fold_date: net daily log-return series}
    for name, wfn in arm_fns.items():
        r = backtest.run_walkforward(name, rets, wfn, uni, run_start, run_end,
                                     window=WINDOW, step=STEP)
        arm_runs[name] = r
        per_fold = {}
        for f in r.folds:
            seg = r.daily_returns.loc[f.date:].iloc[:STEP].copy()
            seg.iloc[0] -= f.turnover * COST * 1e-4
            per_fold[f.date] = seg
        arm_fold_net[name] = per_fold
        m = backtest.BacktestResult(name, r.folds,
                                    r.daily_returns.loc[report_start:])
        mm = m.metrics(cost_bps=COST)
        rows.append({"N": N, "method": name,
                     **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        print(f"  N={N} {name:16s} ret={mm['ann_ret_net']*100:6.2f}%  "
              f"vol={mm['ann_vol_net']*100:6.2f}%  SR={mm['sharpe_net']:+.2f}  "
              f"So={mm['sortino_net']:+.2f}", flush=True)

    all_fold_dates = sorted(arm_fold_net[ARMS[0]])

    def trailing_scores(as_of):
        """Sortino of each arm over the 12 folds strictly before as_of."""
        prior = [d for d in all_fold_dates if d < as_of][-TRAIL_FOLDS:]
        out = {}
        for a in ARMS:
            if not prior:
                out[a] = 0.0
                continue
            net = np.expm1(pd.concat([arm_fold_net[a][d] for d in prior]))
            dn = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2))
            out[a] = float(net.mean() / dn) if dn > 0 else 0.0
        return out

    def rule_fn(rule):
        def fn(R, symbols, as_of):
            scores = trailing_scores(as_of)
            if rule == "R1_fast_wta":
                pick = max(scores, key=scores.get)
                alpha = {a: float(a == pick) for a in ARMS}
            else:                                     # R2 soft blend
                pos = {a: max(s, 0.0) for a, s in scores.items()}
                tot = sum(pos.values())
                alpha = ({a: p / tot for a, p in pos.items()} if tot > 0
                         else {a: 1.0 / len(ARMS) for a in ARMS})
            sel_rows.append({"N": N, "rule": rule, "date": str(as_of.date()),
                             **{a: round(alpha[a], 3) for a in ARMS}})
            w = sum(alpha[a] * np.asarray(arm_fns[a](R, symbols, as_of))
                    for a in ARMS)
            return w
        return fn

    baseline_fns = {
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
    }

    def blend5050(R, symbols, as_of):
        return 0.5 * np.asarray(arm_fns["A1_gmv_tglobal"](R, symbols, as_of)) + \
               0.5 * np.asarray(arm_fns["A3_hrp_gerber"](R, symbols, as_of))

    contenders = {"R1_fast_wta": rule_fn("R1_fast_wta"),
                  "R2_soft_blend": rule_fn("R2_soft_blend"),
                  "BLEND_50_50": blend5050, **baseline_fns}
    runs = {}
    for name, wfn in contenders.items():
        r = backtest.run_walkforward(name, rets, wfn, uni, report_start,
                                     run_end, window=WINDOW, step=STEP)
        runs[name] = r
        mm = r.metrics(cost_bps=COST)
        rows.append({"N": N, "method": name,
                     **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        print(f"  N={N} {name:16s} ret={mm['ann_ret_net']*100:6.2f}%  "
              f"vol={mm['ann_vol_net']*100:6.2f}%  SR={mm['sharpe_net']:+.2f}  "
              f"So={mm['sortino_net']:+.2f}  mdd={mm['max_drawdown']*100:6.1f}%",
              flush=True)

    for cand in ["R1_fast_wta", "R2_soft_blend", "BLEND_50_50"]:
        for b in ["hrp", "gerber_c0.5", "ans"]:
            a_l = runs[cand].fold_losses()
            b_l = runs[b].fold_losses()
            n_ = min(len(a_l), len(b_l))
            dm, p = stats.diebold_mariano(a_l[:n_], b_l[:n_])
            dm_rows.append({"N": N, "method": cand, "vs": b,
                            "dm": round(float(dm), 3), "p": round(float(p), 5)})

pd.DataFrame(rows).to_csv(RESULTS / "arch_v2b_metrics.csv", index=False)
pd.DataFrame(sel_rows).to_csv(RESULTS / "arch_v2b_selection.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "arch_v2b_dm.csv", index=False)
print("\narchitecture v2b done.")
