"""Five-metric completion of the ORIGINAL mcap universes (top-100, top-400).

The audit-rerun tables (main_metrics_gs2022*.csv) predate the sortino_net
metric and lack the HRP-style arms, so the user's 5-metric panel (AnnRet,
AnnVol, Sharpe, Sortino, MDD) cannot be read off them. This script:

  - re-runs A1 (T-global) and A2 (T-asym) from the SAVED audit-rerun
    thresholds (scalar_thresholds_gs2022*.json, c = softplus(raw)) — no
    retraining, bit-identical portfolios to the rerun;
  - adds A3 (HRP on the learned-Gerber correlation), the 50/50 blend, and
    X2 (learned shrinkage toward the learned-Gerber target — the design
    pre-registered in prereg_extensions.py, applied unchanged to these
    universes, single evaluation);
  - re-runs the four baselines (fixed Gerber c=0.5, Ledoit-Wolf, ANS, HRP);
  - reports the full 5-metric panel at {0, 10, 25, 50} bps + DM tests.

Frozen protocol throughout (window 252/step 21, eval 2017+, annual refit
with horizon-aware boundary, fresh Adam lr 0.02 x 40 epochs for X2,
straight-through, gs2022, scale-relative ridge 1e-3, simple-return metrics).
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
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
UNIS = os.environ.get("DG_UNIS", "top100,top400").split(",")
COSTS = [0.0, 10.0, 25.0, 50.0]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index


def softplus(x):
    return math.log1p(math.exp(x))


def load_thresholds(path):
    """JSON raw params -> {module: {year: (c_up, c_down)}}."""
    raw = json.load(open(path))
    out = {"tglobal": {}, "tasym": {}}
    for k, v in raw.items():
        mod, year = k.rsplit("_", 1)
        if "raw" in v:
            out[mod][int(year)] = (softplus(v["raw"]), None)
        else:
            out[mod][int(year)] = (softplus(v["raw_up"]), softplus(v["raw_down"]))
    return out


def gerber_G(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


class ShrinkGerber(nn.Module):
    def __init__(self):
        super().__init__()
        self.thr = GlobalThreshold(0.5)
        self.raw_d = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        return self.thr(), torch.sigmoid(self.raw_d)


def sample_cov_t(Rw):
    Rc = Rw - Rw.mean(0, keepdim=True)
    return Rc.T @ Rc / (Rw.shape[0] - 1)


def train_x2(folds):
    """Identical to prereg_extensions.train_x2 (pre-registered design)."""
    torch.manual_seed(0)
    module = ShrinkGerber()
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
                c, delta = module()
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, c, tau)
                Sig_g = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                Sig = delta * Sig_g + (1.0 - delta) * sample_cov_t(f["Rw"])
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(train)).backward()
            opt.step()
        with torch.no_grad():
            c, delta = module()
            per_year[year] = (float(c), float(delta))
        print(f"  X2 {year}: c={per_year[year][0]:.4f} "
              f"delta={per_year[year][1]:.4f} "
              f"({len(train)} folds, {time.time()-t0:.0f}s)", flush=True)
    return per_year


METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]
OURS = ["A1_gmv_tglobal", "A2_gmv_tasym", "A3_hrp_gerber", "BLEND_50_50",
        "X2_shrink"]
BASE = ["gerber_c0.5", "ledoit_wolf", "ans", "hrp"]
rows, dm_rows, x2_rows = [], [], []

for uname in UNIS:
    n_assets = {"top100": 100, "top400": 400}[uname]
    jpath = RESULTS / ("scalar_thresholds_gs2022.json" if uname == "top100"
                       else "scalar_thresholds_gs2022_top400.json")
    cmaps = load_thresholds(jpath)
    print(f"===== {uname} (N={n_assets}) =====", flush=True)
    _m = {}
    def uni(d, n_assets=n_assets, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe(d, n_assets, px, mem)
        return _m[d]

    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        })

    x2 = train_x2(folds)
    for y, (c, d_) in x2.items():
        x2_rows.append({"universe": uname, "year": y,
                        "c_up": round(c, 4), "delta": round(d_, 4)})

    def gmv_fn(mod):
        def fn(R, symbols, as_of):
            cu, cd = cmaps[mod][as_of.year]
            G, s = gerber_G(R, cu, cd)
            return backtest.gmv_from_cov(np.outer(s, s) * G)
        return fn

    def hrp_fn(mod):
        def fn(R, symbols, as_of):
            cu, cd = cmaps[mod][as_of.year]
            G, _ = gerber_G(R, cu, cd)
            return np.asarray(baselines.hrp_weights(R, corr=G))
        return fn

    a1, a3 = gmv_fn("tglobal"), hrp_fn("tglobal")

    def blend(R, symbols, as_of):
        return 0.5 * np.asarray(a1(R, symbols, as_of)) + \
               0.5 * np.asarray(a3(R, symbols, as_of))

    def x2_fn(R, symbols, as_of):
        c, delta = x2[as_of.year]
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        Sig = delta * (s.unsqueeze(-1) * G * s.unsqueeze(-2)) \
            + (1.0 - delta) * sample_cov_t(Rt)
        return backtest.gmv_from_cov(Sig.numpy())

    contenders = {
        "A1_gmv_tglobal": a1,
        "A2_gmv_tasym": gmv_fn("tasym"),
        "A3_hrp_gerber": a3,
        "BLEND_50_50": blend,
        "X2_shrink": x2_fn,
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

    runs = {}
    for name, wfn in contenders.items():
        r = backtest.run_walkforward(name, rets, wfn, uni, report_start,
                                     run_end, window=WINDOW, step=STEP)
        runs[name] = r
        for cb in COSTS:
            mm = r.metrics(cost_bps=cb)
            rows.append({"universe": uname, "method": name, "cost_bps": cb,
                         **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        mm = r.metrics(cost_bps=10.0)
        print(f"  {uname} {name:15s} ret={mm['ann_ret_net']*100:6.2f}%  "
              f"vol={mm['ann_vol_net']*100:6.2f}%  SR={mm['sharpe_net']:+.2f}  "
              f"So={mm['sortino_net']:+.2f}  mdd={mm['max_drawdown']*100:6.1f}%",
              flush=True)

    for cand in OURS:
        for b in BASE:
            a_l, b_l = runs[cand].fold_losses(), runs[b].fold_losses()
            n_ = min(len(a_l), len(b_l))
            dm, p = stats.diebold_mariano(a_l[:n_], b_l[:n_])
            dm_rows.append({"universe": uname, "method": cand, "vs": b,
                            "dm": round(float(dm), 3), "p": round(float(p), 5)})

pd.DataFrame(rows).to_csv(RESULTS / "top_universe_5metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "top_universe_5metrics_dm.csv", index=False)
pd.DataFrame(x2_rows).to_csv(RESULTS / "top_universe_x2_params.csv", index=False)
print("\ntop-universe 5-metric completion done.")
