"""Architecture v3 — DEVELOPMENT-SET sweep (sector N=30/50).

PROTOCOL DECLARATION: the 2017-26 S&P sector-balanced universes are hereby a
DEVELOPMENT SET. Multiple prior rounds (sector v1, v2, v2b, tier-1/2) have
seen this window; numbers produced here are design evidence, NOT inferential
claims. The single confirmatory test is arch_v3_holdout.py on the untouched
ex-S&P universe. Both scripts are committed BEFORE this sweep runs, and the
winner is chosen by the MECHANICAL rule below with zero human discretion, so
the holdout is fully pre-registered.

Candidates per universe (all covariance from the differentiable learned core;
no retraining — G thresholds from arch_v2b_thresholds_N{N}.csv, X2 (c, delta)
from prereg_extensions_params.csv):

  singles (7): GMV_X2, GMV_G, HRP_single_G, HRP_avg_G, HRP_ward_G,
               HRP_avg_X2, HRP_ward_X2
  fixed blends (9): BL{a}_{x} = a*GMV_X2 + (1-a)*HRP_{x}_G,
                    a in {0.3, 0.5, 0.7}, x in {single, avg, ward}
  learned blend (1): BL_LEARN — alpha = sigmoid(theta) trained by the
      downside decision loss (Sortino-aligned) on w(alpha) =
      alpha*w_GMV_X2 + (1-alpha)*w_HRP_single_G, annual expanding refit,
      fresh Adam lr 0.02 x 40 epochs (the end-to-end learnable combiner).

SELECTION RULE (mechanical, fixed here): per universe, count strictly-won
cells over 4 baselines x 5 metrics (AnnRet up, AnnVol down, Sharpe up,
Sortino up, MDD up i.e. less negative) at 10 bps; winner = max cells,
tie-break = higher net Sharpe. Written to arch_v3_winner.csv, which
arch_v3_holdout.py reads without human intervention.
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.soft_gerber import hard_gerber

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50").split(",")]
COSTS = [0.0, 10.0, 25.0, 50.0]
ALPHAS = [0.3, 0.5, 0.7]
LINKS = ["single", "avg", "ward"]
LINKMAP = {"single": "single", "avg": "average", "ward": "ward"}
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

METRIC_DIR = {"ann_ret_net": 1, "ann_vol_net": -1, "sharpe_net": 1,
              "sortino_net": 1, "max_drawdown": 1}
METRIC_KEYS = list(METRIC_DIR) + ["avg_turnover"]
BASE = ["gerber_c0.5", "ledoit_wolf", "ans", "hrp"]

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index


def make_components(N):
    """Per-universe weight-function builders from SAVED parameters."""
    th = pd.read_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv")
    g = {int(r.year): float(r.c_up)
         for r in th[th.module == "tglobal"].itertuples()}
    xp = pd.read_csv(RESULTS / "prereg_extensions_params.csv")
    xp = xp[(xp.N == N) & (xp.module == "X2_shrink")]
    x2 = {int(r.year): (float(r.c_up), float(r.delta)) for r in xp.itertuples()}

    def g_c(year):
        return g.get(year, 0.5)

    def x2_cd(year):
        return x2.get(year, (0.5, 0.5))

    def gerber_corr(R, c):
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        return hard_gerber(Rt / s, c, "gs2022").numpy(), s.numpy()

    def x2_sigma(R, year):
        c, delta = x2_cd(year)
        G, s = gerber_corr(R, c)
        S = np.cov(R, rowvar=False, ddof=1)
        return delta * (np.outer(s, s) * G) + (1.0 - delta) * S

    def gmv_G(R, symbols, as_of):
        G, s = gerber_corr(R, g_c(as_of.year))
        return backtest.gmv_from_cov(np.outer(s, s) * G)

    def gmv_X2(R, symbols, as_of):
        return backtest.gmv_from_cov(x2_sigma(R, as_of.year))

    def hrp_G(link):
        def fn(R, symbols, as_of):
            G, _ = gerber_corr(R, g_c(as_of.year))
            return np.asarray(baselines.hrp_weights(
                R, corr=G, linkage_method=LINKMAP[link]))
        return fn

    def hrp_X2(link):
        def fn(R, symbols, as_of):
            Sig = x2_sigma(R, as_of.year)
            d = np.sqrt(np.diag(Sig))
            C = Sig / np.outer(d, d)
            np.fill_diagonal(C, 1.0)
            return np.asarray(baselines.hrp_weights(
                R, corr=C, linkage_method=LINKMAP[link]))
        return fn

    return {"gmv_G": gmv_G, "gmv_X2": gmv_X2,
            **{f"hrp_{l}_G": hrp_G(l) for l in LINKS},
            **{f"hrp_{l}_X2": hrp_X2(l) for l in ["avg", "ward"]}}


def train_blend_alpha(folds, comp):
    """BL_LEARN: alpha per year via downside decision loss on fixed
    component weights (annual expanding refit, fresh Adam, 40 epochs)."""
    pre = []
    for f in folds:
        wg = comp["gmv_X2"](f["Rw_np"], f["syms"], f["date"])
        wh = comp["hrp_single_G"](f["Rw_np"], f["syms"], f["date"])
        pre.append({"date": f["date"], "wg": torch.as_tensor(wg),
                    "wh": torch.as_tensor(wh), "Rf": f["Rf"]})
    per_year = {}
    for year in range(2017, LAST_YEAR + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = [p for p in pre
                 if dates[min(dates.get_loc(p["date"]) + STEP,
                              len(dates) - 1)] < cutoff]
        theta = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([theta], lr=0.02)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = 0.0
            for p in train:
                a = torch.sigmoid(theta)
                w = a * p["wg"] + (1 - a) * p["wh"]
                port = p["Rf"] @ w
                loss = loss + (torch.clamp(port, max=0.0) ** 2).mean()
            (loss / len(train)).backward()
            opt.step()
        with torch.no_grad():
            per_year[year] = float(torch.sigmoid(theta))
        print(f"  BL_LEARN {year}: alpha={per_year[year]:.3f}", flush=True)
    return per_year


rows, dom_rows, win_rows = [], [], []

for N in SIZES:
    print(f"===== dev N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    comp = make_components(N)
    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        Rw = rets[syms].iloc[i - WINDOW : i].fillna(0.0).values
        folds.append({"date": dates[i], "syms": syms, "Rw_np": Rw,
                      "Rf": torch.as_tensor(
                          rets[syms].iloc[i : i + STEP].fillna(0.0).values)})
    alpha = train_blend_alpha(
        [f for f in folds if f["date"].year <= LAST_YEAR], comp)

    def blend_fn(a_fixed=None, link="single"):
        def fn(R, symbols, as_of):
            a = alpha[as_of.year] if a_fixed is None else a_fixed
            return a * np.asarray(comp["gmv_X2"](R, symbols, as_of)) + \
                (1 - a) * np.asarray(comp[f"hrp_{link}_G"](R, symbols, as_of))
        return fn

    candidates = {
        "GMV_X2": comp["gmv_X2"], "GMV_G": comp["gmv_G"],
        **{f"HRP_{l}_G": comp[f"hrp_{l}_G"] for l in LINKS},
        **{f"HRP_{l}_X2": comp[f"hrp_{l}_X2"] for l in ["avg", "ward"]},
        **{f"BL{a}_{l}": blend_fn(a, l) for a in ALPHAS for l in LINKS},
        "BL_LEARN": blend_fn(None, "single"),
    }
    baseline_fns = {
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
    }

    fold_dates = [f["date"] for f in folds if f["date"].year <= LAST_YEAR]
    report_start = min(d for d in fold_dates if d.year >= 2017)
    run_end = max(fold_dates) + pd.Timedelta(days=1)

    panel = {}
    for name, wfn in {**candidates, **baseline_fns}.items():
        r = backtest.run_walkforward(name, rets, wfn, uni, report_start,
                                     run_end, window=WINDOW, step=STEP)
        for cb in COSTS:
            mm = r.metrics(cost_bps=cb)
            rows.append({"N": N, "method": name, "cost_bps": cb,
                         **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        panel[name] = r.metrics(cost_bps=10.0)
        mm = panel[name]
        print(f"  N={N} {name:14s} ret={mm['ann_ret_net']*100:6.2f}%  "
              f"vol={mm['ann_vol_net']*100:6.2f}%  SR={mm['sharpe_net']:+.2f}  "
              f"So={mm['sortino_net']:+.2f}  mdd={mm['max_drawdown']*100:6.1f}%",
              flush=True)

    # mechanical selection
    scored = []
    for name in candidates:
        cells, lost = 0, []
        for b in BASE:
            for m, d_ in METRIC_DIR.items():
                if d_ * (panel[name][m] - panel[b][m]) > 0:
                    cells += 1
                else:
                    lost.append(f"{b}:{m}")
        scored.append((cells, panel[name]["sharpe_net"], name, lost))
        dom_rows.append({"N": N, "method": name, "cells_won": cells,
                         "cells_lost": ";".join(lost)})
    scored.sort(key=lambda t: (-t[0], -t[1]))
    cells, sr, wname, lost = scored[0]
    win_rows.append({"N": N, "winner": wname, "cells_won": cells,
                     "sharpe_net": round(float(sr), 5),
                     "cells_lost": ";".join(lost)})
    print(f"  >>> N={N} WINNER: {wname} ({cells}/20 cells, SR={sr:+.3f})",
          flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "arch_v3_dev_metrics.csv", index=False)
pd.DataFrame(dom_rows).to_csv(RESULTS / "arch_v3_dev_domination.csv", index=False)
pd.DataFrame(win_rows).to_csv(RESULTS / "arch_v3_winner.csv", index=False)
print("\narch v3 dev sweep done.")
