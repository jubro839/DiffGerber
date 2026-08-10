"""End-to-end demo on synthetic data — no database, no vendor data, ~1 minute.

Exercises the exact pipeline used for the paper's real-data results: a soft
Gerber layer with the straight-through estimator, a learned global threshold
trained through the differentiable minimum-variance layer on realized
portfolio variance, annual expanding refits with a horizon-aware boundary, and
the same walk-forward engine and metrics (net of transaction costs). Only the
data source differs.

The synthetic returns follow a one-factor model with Student-t innovations, so
the dependence lives partly in the tails: the setting where a co-movement
threshold has something to decide. Because the generator is fixed, the printed
numbers are reproducible on any machine.

    python experiments/demo_synthetic.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
SEED, N_ASSETS, N_DAYS = 0, 25, 2200
WINDOW, STEP, EPOCHS = 252, 21, 20
DF_TAIL = 4.0                      # Student-t degrees of freedom
COST_BPS = 10.0


def synthetic_panel():
    """One-factor returns with heavy-tailed innovations, as a wide frame."""
    rng = np.random.default_rng(SEED)
    beta = rng.uniform(0.4, 1.4, N_ASSETS)
    idio_vol = rng.uniform(0.008, 0.022, N_ASSETS)
    factor = rng.standard_t(DF_TAIL, N_DAYS)[:, None] * 0.009
    idio = rng.standard_t(DF_TAIL, (N_DAYS, N_ASSETS)) * idio_vol
    rets = factor * beta + idio
    dates = pd.bdate_range("2016-01-01", periods=N_DAYS)
    cols = [f"A{i:02d}" for i in range(N_ASSETS)]
    return pd.DataFrame(rets, index=dates, columns=cols)


def build_folds(rets, dates):
    folds = []
    for i in range(WINDOW, len(dates) - STEP, STEP):
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets.iloc[i - WINDOW:i].values),
            "Rf": torch.as_tensor(rets.iloc[i:i + STEP].values),
        })
    return folds


def train_threshold(folds, dates, years):
    """Annual expanding refit, fresh optimizer each year, horizon-aware
    boundary — identical discipline to the paper's protocol."""
    torch.manual_seed(SEED)
    module = GlobalThreshold(0.5)
    layer = SoftGerber("gs2022", straight_through=True)
    per_year = {}
    for year in years:
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = [f for f in folds
                 if dates[min(dates.get_loc(f["date"]) + STEP, len(dates) - 1)]
                 < cutoff]
        if not train:
            per_year[year] = 0.5
            continue
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
        with torch.no_grad():
            per_year[year] = float(module())
        print(f"  refit {year}: learned c = {per_year[year]:.4f} "
              f"({len(train)} folds)", flush=True)
    return per_year


def main():
    rets = synthetic_panel()
    dates = rets.index
    symbols = list(rets.columns)
    folds = build_folds(rets, dates)

    eval_years = sorted({d.year for d in dates})[2:]      # leave 2 warm-up yrs
    eval_start = min(d for d in dates if d.year == eval_years[0])
    eval_start = dates[max(dates.get_loc(eval_start), WINDOW)]
    eval_end = dates[-1]

    print(f"synthetic panel: {N_ASSETS} assets x {N_DAYS} days, "
          f"Student-t({DF_TAIL:.0f}) innovations")
    print(f"training the threshold ({EPOCHS} epochs per refit):")
    cmap = train_threshold(folds, dates, eval_years)

    def uni(_as_of):
        return symbols

    def learned_fn(R, _symbols, as_of):
        c = cmap.get(as_of.year, 0.5)
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        return backtest.gmv_from_cov(
            (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())

    contenders = {
        "DiffGerber (learned c)": learned_fn,
        "Gerber c=0.5 (published)": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "Ledoit-Wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "sample covariance": backtest.covariance_weight_fn(baselines.sample_cov),
    }

    print(f"\nwalk-forward {eval_start.date()} to {eval_end.date()}, "
          f"net of {COST_BPS:.0f} bp:")
    print(f"  {'method':26s} {'Ret':>7s} {'Vol':>7s} {'Sharpe':>7s} "
          f"{'Sortino':>8s} {'MDD':>7s}")
    for name, fn in contenders.items():
        r = backtest.run_walkforward(name, rets, fn, uni, eval_start, eval_end,
                                     window=WINDOW, step=STEP)
        m = r.metrics(cost_bps=COST_BPS)
        print(f"  {name:26s} {m['ann_ret_net']*100:6.2f}% "
              f"{m['ann_vol_net']*100:6.2f}% {m['sharpe_net']:7.2f} "
              f"{m['sortino_net']:8.2f} {m['max_drawdown']*100:6.1f}%")

    c_late = np.mean([cmap[y] for y in eval_years[2:]]) if len(eval_years) > 2 \
        else np.mean(list(cmap.values()))
    print(f"\nWhat to look at: the learned threshold settles near "
          f"{c_late:.2f}, below the published 0.5.")
    print("That direction is the paper's tail mechanism (Section 6.7): heavy")
    print("tails inflate each asset's standard deviation, so a threshold in")
    print("units of sigma captures fewer exceedances and the optimum falls.")
    print("\nThe performance table above is a pipeline check, not evidence: 25")
    print("assets, one generator draw and 20 epochs cannot separate estimators")
    print("(the spread here is well inside sampling noise). The paper's")
    print("comparisons use point-in-time equity universes with significance")
    print("tests; see README.md for how to reproduce them.")


if __name__ == "__main__":
    main()
