"""Forward/backward ablation: what is actually differentiable, and what is not.

The Gerber statistic is built from indicators, so the exact objective
L(c) = realized variance of the GMV portfolio built from the hard statistic
is piecewise constant in c: its true gradient is zero almost everywhere. Any
gradient-based method therefore optimizes something other than the exact
gradient. This script separates the two design choices that are usually
conflated and measures each.

  FORWARD  which statistic is actually evaluated (and deployed)
             hard  — the published ternary statistic
             soft  — the tempered three-state relaxation at temperature tau
  BACKWARD what supplies the search direction
             surrogate — d/dc of the soft relaxation (tempered softmax)
             finite    — central difference of the EXACT hard objective,
                         (L(c+h) - L(c-h)) / 2h; no relaxation anywhere

Variants evaluated (top-100, frozen protocol):
  A  STE          forward hard,  backward surrogate   (the method in the paper)
  B  soft/soft    forward soft,  backward surrogate, deployed soft
  C  soft/hard    forward soft,  backward surrogate, deployed hard
  D  finite-diff  forward hard,  backward finite difference of the exact loss

D also records how often the finite-difference gradient is exactly zero, which
is the quantitative reason a relaxation is needed at all. Every variant reports
its per-year learned threshold alongside out-of-sample performance, so the
optimization path is visible and not just its end point.

Frozen protocol throughout: window 252 / step 21, evaluation 2017+, annual
expanding refit with horizon-aware boundary, fresh Adam at lr 0.02 for 40
steps, tau annealed 0.2 -> 0.02, gs2022 normalization, scale-relative ridge
1e-3, metrics net of 10 bps.
"""

import math
import os
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
from diffgerber.soft_gerber import SoftGerber, hard_gerber, soft_exceedance, \
    gerber_from_ternary

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
TOPN = int(os.environ.get("DG_TOPN", 100))
FD_H = 0.05                    # finite-difference step, in units of c
LR = 0.02
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index

_uni_cache = {}


def uni(d):
    if d not in _uni_cache:
        _uni_cache[d] = data.pit_universe(d, TOPN, px, mem)
    return _uni_cache[d]


folds = []
for i in range(WINDOW, len(dates) - 1, STEP):
    syms = uni(dates[i])
    folds.append({
        "date": dates[i],
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i:i + STEP].fillna(0.0).values),
    })
fold_dates = [f["date"] for f in folds if f["date"].year <= LAST_YEAR]
report_start = min(d for d in fold_dates if d.year >= 2017)
run_end = max(fold_dates) + pd.Timedelta(days=1)


def train_folds(year):
    cutoff = pd.Timestamp(f"{year}-01-01")
    return [f for f in folds
            if dates[min(dates.get_loc(f["date"]) + STEP, len(dates) - 1)] < cutoff]


def softplus(x):
    return math.log1p(math.exp(x)) if x < 30 else x


def inv_softplus(y):
    return math.log(math.expm1(y))


def fold_loss_soft(f, c, tau, straight_through, denominator_grad=True):
    """Differentiable loss for one fold under the tempered relaxation."""
    layer = SoftGerber("gs2022", straight_through=straight_through,
                       denominator_grad=denominator_grad)
    s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
    G = layer(f["Rw"] / s, c, tau)
    Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
    return decision_loss(gmv_closed_form(Sig), f["Rf"])


def hard_loss_value(f, c_val):
    """Exact loss of the deployed hard statistic at threshold c (no grad)."""
    with torch.no_grad():
        s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(f["Rw"] / s, c_val, "gs2022")
        Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
        return float(decision_loss(gmv_closed_form(Sig), f["Rf"]))


# mean wall-clock seconds per refit, per variant: the manuscript
# compares the surrogate's cost against the finite difference's
REFIT_SECONDS = {}


def train_gradient(straight_through, tag, denominator_grad=True):
    """Variants A/B/C: Adam on the surrogate gradient.

    Protocol detail that must match the headline runs: the threshold
    parameter persists across refit years (each year starts from the previous
    year's value) while the optimizer is re-instantiated, so later years do
    not accumulate optimizer state.
    """
    torch.manual_seed(0)
    per_year, path_rows, secs = {}, [], []
    raw = torch.tensor(inv_softplus(0.5), requires_grad=True)
    for year in range(2017, LAST_YEAR + 1):
        opt = torch.optim.Adam([raw], lr=LR)
        tr = train_folds(year)
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            c = torch.nn.functional.softplus(raw)
            loss = sum(fold_loss_soft(f, c, tau, straight_through,
                                      denominator_grad) for f in tr)
            (loss / len(tr)).backward()
            opt.step()
            if year == LAST_YEAR:
                path_rows.append({"variant": tag, "step": ep,
                                  "c": round(softplus(float(raw.detach())), 5)})
        per_year[year] = softplus(float(raw.detach()))
        secs.append(time.time() - t0)
        print(f"  {tag} {year}: c={per_year[year]:.4f} "
              f"({len(tr)} folds, {secs[-1]:.0f}s)", flush=True)
    REFIT_SECONDS[tag] = float(np.mean(secs))
    return per_year, path_rows


def train_finite_difference(tag="D_finite_diff"):
    """Variant D: Adam driven by a central difference of the EXACT loss."""
    per_year, path_rows, secs = {}, [], []
    zero_grad_steps = total_steps = 0
    raw = torch.tensor(inv_softplus(0.5))
    raw.requires_grad_(True)
    for year in range(2017, LAST_YEAR + 1):
        opt = torch.optim.Adam([raw], lr=LR)
        tr = train_folds(year)
        t0 = time.time()
        for ep in range(EPOCHS):
            c_val = softplus(float(raw))
            lo = np.mean([hard_loss_value(f, max(c_val - FD_H, 1e-4)) for f in tr])
            hi = np.mean([hard_loss_value(f, c_val + FD_H) for f in tr])
            dL_dc = (hi - lo) / (2 * FD_H)
            total_steps += 1
            if dL_dc == 0.0:
                zero_grad_steps += 1
            # chain through the softplus reparameterization
            dc_draw = 1.0 / (1.0 + math.exp(-float(raw)))
            opt.zero_grad()
            raw.grad = torch.tensor(dL_dc * dc_draw)
            opt.step()
            if year == LAST_YEAR:
                path_rows.append({"variant": tag, "step": ep,
                                  "c": round(softplus(float(raw)), 5)})
        per_year[year] = softplus(float(raw.detach()))
        secs.append(time.time() - t0)
        print(f"  {tag} {year}: c={per_year[year]:.4f} "
              f"({len(tr)} folds, {secs[-1]:.0f}s)", flush=True)
    REFIT_SECONDS[tag] = float(np.mean(secs))
    print(f"  finite-difference gradient was exactly zero in "
          f"{zero_grad_steps}/{total_steps} steps", flush=True)
    return per_year, path_rows, zero_grad_steps, total_steps


def weight_fn_hard(cmap):
    def fn(R, symbols, as_of):
        c = cmap.get(as_of.year, 0.5)
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        return backtest.gmv_from_cov((s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
    return fn


def weight_fn_soft(cmap, tau=0.02):
    """Deploy the RELAXED statistic itself (variant B)."""
    def fn(R, symbols, as_of):
        c = cmap.get(as_of.year, 0.5)
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        z = Rt / s
        u, d, n = soft_exceedance(z, torch.tensor(c), tau)
        G = gerber_from_ternary(u - d, n=n, normalization="gs2022")
        return backtest.gmv_from_cov((s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
    return fn


rows, cpath_rows, step_rows = [], [], []

print(f"=== forward/backward ablation, top-{TOPN} ===", flush=True)
c_ste, path_a = train_gradient(True, "A_ste")
c_soft, path_b = train_gradient(False, "B_soft")
c_fd, path_d, n_zero, n_tot = train_finite_difference()
# E: same construction as A but the denominator's n path severed, so the
# gradient sees only the numerator channel. Measures whether the denominator
# gradient -- 32-62% of the magnitude per gradient_decomposition.py -- is
# load-bearing for where the threshold ends up.
c_num, path_e = train_gradient(True, "E_numonly", denominator_grad=False)

VARIANTS = {
    "A_ste_hard":       ("hard", "surrogate", "hard", weight_fn_hard(c_ste), c_ste),
    "E_num_only":       ("hard", "surr-num",  "hard", weight_fn_hard(c_num), c_num),
    "B_soft_soft":      ("soft", "surrogate", "soft", weight_fn_soft(c_soft), c_soft),
    "C_soft_deployhard": ("soft", "surrogate", "hard", weight_fn_hard(c_soft), c_soft),
    "D_finite_diff":    ("hard", "finite",    "hard", weight_fn_hard(c_fd), c_fd),
}

for name, (fwd, bwd, deploy, wfn, cmap) in VARIANTS.items():
    r = backtest.run_walkforward(name, rets, wfn, uni, report_start, run_end,
                                 window=WINDOW, step=STEP)
    m = r.metrics(cost_bps=10.0)
    rows.append({"universe": f"top{TOPN}", "variant": name, "forward": fwd,
                 "backward": bwd, "deployed": deploy,
                 "c_mean_2019plus": round(float(np.mean(
                     [v for y, v in cmap.items() if y >= 2019]
                     or list(cmap.values()))), 4),
                 "c_final": round(float(cmap[LAST_YEAR]), 4),
                 **{k: round(float(m[k]), 5) for k in METRIC_KEYS}})
    for y, v in cmap.items():
        cpath_rows.append({"variant": name, "year": y, "c": round(float(v), 4)})
    print(f"  {name:18s} c={rows[-1]['c_mean_2019plus']:.3f}  "
          f"vol={m['ann_vol_net']*100:6.3f}%  SR={m['sharpe_net']:+.3f}  "
          f"mdd={m['max_drawdown']*100:6.1f}%", flush=True)

rows.append({"universe": f"top{TOPN}", "variant": "FD_zero_gradient_rate",
             "forward": "hard", "backward": "finite", "deployed": "-",
             "c_mean_2019plus": round(n_zero / max(n_tot, 1), 4),
             "c_final": n_tot})
step_rows = path_a + path_b + path_d + path_e

pd.DataFrame(rows).to_csv(RESULTS / "ablation_forward_backward.csv", index=False)
pd.DataFrame(cpath_rows).to_csv(RESULTS / "ablation_fb_thresholds.csv", index=False)
pd.DataFrame(step_rows).to_csv(RESULTS / "ablation_fb_optpath.csv", index=False)
pd.DataFrame([{"variant": k, "mean_seconds_per_refit": round(v, 2)}
              for k, v in sorted(REFIT_SECONDS.items())]).to_csv(
    RESULTS / "ablation_fb_timing.csv", index=False)
print("\nforward/backward ablation done.")
