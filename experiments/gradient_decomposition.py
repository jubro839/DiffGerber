"""How much of dL/dc arrives through the Gerber DENOMINATOR?

The GS2022 statistic is a quotient: numerator = concordant-minus-discordant
counts built from the ternary signal x, denominator = the share of periods in
which at least one of the pair is active, built from the neutral indicator n.
Our straight-through construction gives BOTH channels a soft gradient
(soft_gerber.py: the STE is applied to x and to n), so the threshold's
gradient decomposes exactly, by the chain rule, into a numerator path and a
denominator path:

    dL/dc = g_num (via x) + g_den (via n)

This script measures the two on real folds -- every evaluation fold of the
top-100 universe, at the published threshold (c = 0.5), at the recovered
optimum (c = 1.0), and at both ends of the temperature anneal. g_num is
computed by severing the n path (SoftGerber(denominator_grad=False)); g_den
follows by subtraction and is cross-checked against an explicit x-detached
pass at the first fold.

Questions the paper needs answered with numbers:
  1. What share of the gradient magnitude flows through the denominator?
  2. Do the two paths oppose each other (they should: raising c shrinks the
     numerator's signal but also shrinks the denominator, which pushes |G|
     the other way)?
  3. Would a numerator-only gradient ever point descent the wrong way --
     per fold, and in the aggregate step actually taken by the optimizer?
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import data
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import (SoftGerber, gerber_from_ternary,
                                    hard_ternary, soft_exceedance)

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
TOPN = int(os.environ.get("DG_TOPN", 100))
CS = [0.5, 1.0]                  # the published default and the recovered optimum
TAUS = [0.2, 0.02]               # both ends of the annealing schedule
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index

_m = {}
def uni(d):
    if d not in _m:
        _m[d] = data.pit_universe(d, TOPN, px, mem)
    return _m[d]

folds = []
# len(dates) - STEP: only folds whose full 21-day horizon lies inside the
# sample, i.e. the same 114 evaluation rebalances the paper reports on
for i in range(WINDOW, len(dates) - STEP, STEP):
    if dates[i].year < 2017:
        continue
    syms = uni(dates[i])
    folds.append({
        "date": dates[i],
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i:i + STEP].fillna(0.0).values),
    })
print(f"{len(folds)} folds, top-{TOPN}")


def fold_grad(f, c_val, tau, layer):
    c = torch.tensor(c_val, requires_grad=True)
    s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
    G = layer(f["Rw"] / s, c, tau)
    Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
    loss = decision_loss(gmv_closed_form(Sig), f["Rf"])
    return float(torch.autograd.grad(loss, c)[0])


def fold_grad_den_only(f, c_val, tau):
    """Explicit cross-check: x carries hard values with NO soft path."""
    c = torch.tensor(c_val, requires_grad=True)
    s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
    z = f["Rw"] / s
    _, _, n = soft_exceedance(z, c, tau)
    x_hard = hard_ternary(z, c)
    n_hard = (x_hard == 0).to(z.dtype)
    n = n + (n_hard - n).detach()
    G = gerber_from_ternary(x_hard.detach(), n=n, normalization="gs2022")
    Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
    loss = decision_loss(gmv_closed_form(Sig), f["Rf"])
    return float(torch.autograd.grad(loss, c)[0])


full_layer = SoftGerber("gs2022", straight_through=True)
num_layer = SoftGerber("gs2022", straight_through=True, denominator_grad=False)

rows = []
for tau in TAUS:
    for c_val in CS:
        g_full = np.array([fold_grad(f, c_val, tau, full_layer) for f in folds])
        g_num = np.array([fold_grad(f, c_val, tau, num_layer) for f in folds])
        g_den = g_full - g_num                      # exact by chain-rule linearity

        # cross-check the subtraction against the explicit x-detached pass
        explicit = fold_grad_den_only(folds[0], c_val, tau)
        assert abs(explicit - g_den[0]) < 1e-9 * max(1.0, abs(explicit)), \
            (explicit, g_den[0])

        share = np.abs(g_den) / (np.abs(g_num) + np.abs(g_den))
        oppose = float(np.mean(np.sign(g_den) == -np.sign(g_num)))
        flip = float(np.mean(np.sign(g_num) != np.sign(g_full)))
        agg_flip = int(np.sign(g_num.mean()) != np.sign(g_full.mean()))
        rows.append({
            "universe": f"top{TOPN}", "c": c_val, "tau": tau,
            "n_folds": len(folds),
            "den_share_mean": round(float(share.mean()), 4),
            "den_share_median": round(float(np.median(share)), 4),
            "opposing_sign_frac": round(oppose, 4),
            "fold_sign_flip_frac": round(flip, 4),
            "aggregate_sign_flip": agg_flip,
            "g_full_mean": float(g_full.mean()),
            "g_num_mean": float(g_num.mean()),
            "g_den_mean": float(g_den.mean()),
        })
        r = rows[-1]
        print(f"  c={c_val} tau={tau:<5} den share {r['den_share_mean']*100:5.1f}% "
              f"(median {r['den_share_median']*100:5.1f}%)  opposing "
              f"{r['opposing_sign_frac']*100:5.1f}%  fold flips "
              f"{r['fold_sign_flip_frac']*100:5.1f}%  aggregate flip: "
              f"{'YES' if agg_flip else 'no'}", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "gradient_decomposition.csv", index=False)
print("\ngradient decomposition done.")
