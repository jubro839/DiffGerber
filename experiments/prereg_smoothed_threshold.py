"""PRE-REGISTERED: does penalizing threshold movement recover the headroom?

Committed before execution. Results enter the record as they come.

MOTIVATION (stated before running, from threshold_ceiling.py). On the
sector-balanced universes the ex-post best fixed threshold is c = 1.5 at
N = 30, 50 and 100 -- a CONSTANT in every case -- and it beats the published
default by 0.10-0.16 Sharpe. Our annual refit captures 61-64% of that at
N = 30 and 50 but is actually WORSE than the default at N = 100 (0.341
against 0.384), despite settling near the right level. The diagnosis this
suggests is that the threshold is being moved too often rather than moved to
the wrong place: year-to-year variation in c costs more than it earns.

HYPOTHESIS. Penalizing year-over-year movement of the threshold recovers part
of the gap, and the gain is largest where the unsmoothed procedure fails
worst, i.e. it should increase with N.

DESIGN. Identical to the frozen protocol in every respect except one added
term in the training objective at refit year t:

    L_total = L_decision + gamma * stopgrad(L_decision) * (c_t - c_{t-1})^2

c_{t-1} is the previous refit's converged threshold, held constant. The
penalty is scaled by the current loss so gamma is dimensionless: gamma = 1
means a one-unit move in the threshold costs as much as doubling the decision
loss, so a typical 0.2 move costs 4% of the loss -- a mild brake, not a
freeze. We fix GAMMA = 1.0 on that reasoning, before seeing any result, and
do not tune it. The first refit year has no predecessor and is unpenalized.

REPORTING RULE. Every universe N in {20, 30, 40, 50, 70, 100} is reported
whether or not smoothing helps, alongside the unsmoothed learned threshold,
the published default, and the ex-post oracle over a fixed grid, so that the
fraction of available headroom captured is visible at each N. No universe is
dropped and no gamma is retried. If this works, the single confirmatory test
is the ex-S&P holdout, as in the earlier architecture round.
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "20,30,40,50,70,100").split(",")]
GAMMA = 1.0                      # fixed a priori; see docstring
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index
fold_dates = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)
              if dates[i].year <= LAST_YEAR]
report_start = min(d for d in fold_dates if d.year >= 2017)
run_end = max(fold_dates) + pd.Timedelta(days=1)


def hard_fn(cmap_or_const):
    """Weight function for a hard Gerber GMV portfolio."""
    def fn(R, symbols, as_of):
        c = (cmap_or_const if np.isscalar(cmap_or_const)
             else cmap_or_const.get(as_of.year, 0.5))
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022")
        return backtest.gmv_from_cov(
            (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
    return fn


def train(folds, smooth):
    """Frozen protocol; `smooth` adds the movement penalty."""
    torch.manual_seed(0)
    module = GlobalThreshold(0.5)
    layer = SoftGerber("gs2022", straight_through=True)
    per_year, c_prev = {}, None
    for year in range(2017, LAST_YEAR + 1):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        tr = [f for f in folds
              if dates[min(dates.get_loc(f["date"]) + STEP,
                           len(dates) - 1)] < cutoff]
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            c = module()
            for f in tr:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, c, tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            loss = loss / len(tr)
            if smooth and c_prev is not None:
                loss = loss + GAMMA * loss.detach() * (c - c_prev) ** 2
            loss.backward()
            opt.step()
        with torch.no_grad():
            per_year[year] = float(module())
        c_prev = per_year[year]
        print(f"    {year}: c={per_year[year]:.4f} ({len(tr)} folds, "
              f"{time.time()-t0:.0f}s)", flush=True)
    return per_year


rows, path_rows = [], []
for N in SIZES:
    print(f"===== sector N={N} =====", flush=True)
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
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i:i + STEP].fillna(0.0).values),
        })

    print("  [unsmoothed]", flush=True)
    c_plain = train(folds, smooth=False)
    print("  [smoothed]", flush=True)
    c_smooth = train(folds, smooth=True)
    for y in c_plain:
        path_rows.append({"N": N, "year": y,
                          "c_unsmoothed": round(c_plain[y], 4),
                          "c_smoothed": round(c_smooth[y], 4)})

    variants = {"learned_unsmoothed": hard_fn(c_plain),
                "learned_smoothed": hard_fn(c_smooth),
                "fixed_c0.5": hard_fn(0.5)}
    for c in GRID:
        variants[f"grid_c{c}"] = hard_fn(c)

    for name, fn in variants.items():
        r = backtest.run_walkforward(name, rets, fn, uni, report_start,
                                     run_end, window=WINDOW, step=STEP)
        m = r.metrics(cost_bps=10.0)
        rows.append({"N": N, "rule": name,
                     "ann_ret_net": round(float(m["ann_ret_net"]), 5),
                     "ann_vol_net": round(float(m["ann_vol_net"]), 5),
                     "sharpe_net": round(float(m["sharpe_net"]), 5),
                     "sortino_net": round(float(m["sortino_net"]), 5),
                     "max_drawdown": round(float(m["max_drawdown"]), 5),
                     "avg_turnover": round(float(m["avg_turnover"]), 5)})
        if name in ("learned_unsmoothed", "learned_smoothed", "fixed_c0.5"):
            print(f"  {name:20s} vol={m['ann_vol_net']*100:6.3f}%  "
                  f"SR={m['sharpe_net']:+.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "prereg_smoothed_metrics.csv", index=False)
pd.DataFrame(path_rows).to_csv(RESULTS / "prereg_smoothed_paths.csv", index=False)

print("\n=== headroom captured, by N ===")
print(f"{'N':>5s}{'default':>9s}{'unsmooth':>10s}{'smoothed':>10s}"
      f"{'oracle':>9s}{'captured_un':>13s}{'captured_sm':>13s}")
for N in SIZES:
    s = df[df.N == N].set_index("rule")
    g = s[s.index.str.startswith("grid_")]
    orc = g.sharpe_net.max()
    d0 = s.loc["fixed_c0.5", "sharpe_net"]
    un = s.loc["learned_unsmoothed", "sharpe_net"]
    sm = s.loc["learned_smoothed", "sharpe_net"]
    denom = orc - d0
    fu = (un - d0) / denom * 100 if denom > 0 else float("nan")
    fs = (sm - d0) / denom * 100 if denom > 0 else float("nan")
    print(f"{N:>5d}{d0:9.3f}{un:10.3f}{sm:10.3f}{orc:9.3f}"
          f"{fu:12.0f}%{fs:12.0f}%")
print("\npre-registered smoothing experiment done.")
