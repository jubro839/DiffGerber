"""Pre-registered Tier-2 extensions, evaluated ONCE. Registered before any
result is seen; both outcomes will be reported as-is.

X1  Turnover-penalized threshold learning (Hautsch-Voigt 2019 motivation).
    T-global GMV; training loss = realized variance + LAM * ||w_k - w_{k-1}||_1
    with LAM = 0.001/21 (10 bps amortized over the 21-day horizon) — a single
    fixed value, NOT tuned. Folds processed in date order; the previous fold's
    weights are detached (gradient truncation); weights aligned on the symbol
    union when the universe changes. Deployment identical to A1 (GMV on the
    hard learned-c Gerber), so any change comes through the learned c alone.

X2  Learned shrinkage toward a learned Gerber target (Ledoit-Wolf 2003 target
    literature): Sigma(c, delta) = delta * D^1/2 G(c) D^1/2 + (1-delta) * S_sample,
    delta = sigmoid(raw) init 0.5, c init 0.5, (c, delta) trained jointly with
    the same variance decision loss. Targets the MDD column, where shrinkage
    estimators have been the persistent winners. Deployment: hard Gerber at the
    learned c blended with the sample covariance at the learned delta.

Everything else is the frozen protocol: window 252 / step 21, eval 2017+,
annual refit with horizon-aware boundary, fresh Adam lr 0.02 x 40 epochs,
straight-through estimator, gs2022 normalization, scale-relative ridge 1e-3,
sector-balanced universes N in {30, 50, 100}, metrics net of 10 bps, plus a
{0, 10, 25, 50} bps cost curve and DM tests vs the four baselines.
"""

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
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,100").split(",")]
LAM = 0.001 / 21.0          # X1 penalty: 10 bps amortized per day, fixed
COSTS = [0.0, 10.0, 25.0, 50.0]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index


class ShrinkGerber(nn.Module):
    """c = softplus(raw_c) via GlobalThreshold; delta = sigmoid(raw_d)."""

    def __init__(self):
        super().__init__()
        self.thr = GlobalThreshold(0.5)
        self.raw_d = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

    def forward(self):
        return self.thr(), torch.sigmoid(self.raw_d)


def sample_cov_t(Rw):
    Rc = Rw - Rw.mean(0, keepdim=True)
    return Rc.T @ Rc / (Rw.shape[0] - 1)


def train_x1(folds):
    torch.manual_seed(0)
    module = GlobalThreshold(0.5)
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
            prev_w, prev_syms = None, None
            for f in train:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, module(), tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                w = gmv_closed_form(Sig)
                loss = loss + decision_loss(w, f["Rf"])
                if prev_w is not None:
                    union = list(dict.fromkeys(prev_syms + f["syms"]))
                    iu = {sym: i for i, sym in enumerate(union)}
                    cv = torch.zeros(len(union)).index_add(
                        0, torch.tensor([iu[x] for x in f["syms"]]), w)
                    pv = torch.zeros(len(union)).index_add(
                        0, torch.tensor([iu[x] for x in prev_syms]), prev_w)
                    loss = loss + LAM * (cv - pv).abs().sum()
                prev_w, prev_syms = w.detach(), f["syms"]
            (loss / len(train)).backward()
            opt.step()
        with torch.no_grad():
            per_year[year] = float(module())
        print(f"  X1 {year}: c={per_year[year]:.4f} "
              f"({len(train)} folds, {time.time()-t0:.0f}s)", flush=True)
    return per_year


def train_x2(folds):
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
rows, dm_rows, p_rows = [], [], []

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
            "date": dates[i], "syms": syms,
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        })

    x1 = train_x1(folds)
    x2 = train_x2(folds)
    for y, c in x1.items():
        p_rows.append({"N": N, "module": "X1_turnpen", "year": y,
                       "c_up": round(c, 4), "delta": None})
    for y, (c, d_) in x2.items():
        p_rows.append({"N": N, "module": "X2_shrink", "year": y,
                       "c_up": round(c, 4), "delta": round(d_, 4)})

    def x1_fn(R, symbols, as_of):
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, x1[as_of.year], "gs2022")
        return backtest.gmv_from_cov(
            (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())

    def x2_fn(R, symbols, as_of):
        c, delta = x2[as_of.year]
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        Sig = delta * (s.unsqueeze(-1) * G * s.unsqueeze(-2)) \
            + (1.0 - delta) * sample_cov_t(Rt)
        return backtest.gmv_from_cov(Sig.numpy())

    contenders = {
        "X1_turnpen": x1_fn,
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
            rows.append({"N": N, "method": name, "cost_bps": cb,
                         **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        mm = r.metrics(cost_bps=10.0)
        print(f"  N={N} {name:12s} ret={mm['ann_ret_net']*100:6.2f}%  "
              f"vol={mm['ann_vol_net']*100:6.2f}%  SR={mm['sharpe_net']:+.2f}  "
              f"So={mm['sortino_net']:+.2f}  mdd={mm['max_drawdown']*100:6.1f}%",
              flush=True)

    for cand in ["X1_turnpen", "X2_shrink"]:
        for b in ["gerber_c0.5", "ledoit_wolf", "ans", "hrp"]:
            a_l, b_l = runs[cand].fold_losses(), runs[b].fold_losses()
            n_ = min(len(a_l), len(b_l))
            dm, p = stats.diebold_mariano(a_l[:n_], b_l[:n_])
            dm_rows.append({"N": N, "method": cand, "vs": b,
                            "dm": round(float(dm), 3), "p": round(float(p), 5)})

pd.DataFrame(rows).to_csv(RESULTS / "prereg_extensions_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "prereg_extensions_dm.csv", index=False)
pd.DataFrame(p_rows).to_csv(RESULTS / "prereg_extensions_params.csv", index=False)
print("\npre-registered extensions done.")
