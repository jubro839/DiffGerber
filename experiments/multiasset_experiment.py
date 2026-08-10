"""Multi-asset universes: the asset mix where the original study lives.

The Gerber statistic was published on a multi-asset universe, not on single
asset-class equities, and equal weighting is a categorically different
benchmark there: with bonds at a quarter of equity volatility, any covariance
estimator produces a far lower-risk portfolio than 1/N. This experiment runs
the frozen protocol on two fixed ETF universes --

    ma9   SPY IWM EFA EEM AGG HYG TIP GLD VNQ
          (US large/small, intl developed/EM, aggregate bonds, high yield,
           TIPS, gold, REITs -- the spirit of the original nine-asset mix)
    ma12  ma9 + TLT LQD DBC (long Treasuries, IG credit, commodities)

-- and measures, ceiling first, how much a threshold can add at all:

    fixed grid c in {0.25..2.25}  the constant-threshold ceiling (ex post)
    learned T-global              annual expanding refit, warm start
    LW / ANS / HRP / EW / sample  the usual comparators

The mechanism map predicts the outcome direction: cross-asset correlations
are low, and low ambient correlation has so far meant the learned threshold
stays near the published default (the ex-S&P holdout pattern). The point of
the run is to measure that, not to assume it.

Outputs: multiasset_metrics.csv, multiasset_dm.csv, multiasset_thresholds.csv
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
WINDOW, STEP, EPOCHS = 252, 21, 40
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
UNIVERSES = {
    "ma9": ["SPY", "IWM", "EFA", "EEM", "AGG", "HYG", "TIP", "GLD", "VNQ"],
    "ma12": ["SPY", "IWM", "EFA", "EEM", "AGG", "HYG", "TIP", "GLD", "VNQ",
             "TLT", "LQD", "DBC"],
}
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

ALL = sorted(set(s for u in UNIVERSES.values() for s in u))
px = data.prices_total_return("2015-01-01", "2026-07-28", symbols=ALL)
rets = data.log_returns(px)
dates = rets.index
print(f"panel: {px.shape}, {dates.min().date()}..{dates.max().date()}")


def train_tglobal(folds, tag):
    """Frozen protocol: warm-started module, fresh Adam per annual refit."""
    torch.manual_seed(0)
    module = GlobalThreshold(0.5)
    layer = SoftGerber("gs2022", straight_through=True)
    learned = {}
    for year in range(2017, 2027):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        tr = [f for f in folds
              if dates[min(dates.get_loc(f["date"]) + STEP,
                           len(dates) - 1)] < cutoff]
        if not tr:
            learned[year] = 0.5
            continue
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in tr:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, module(), tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(tr)).backward()
            opt.step()
        learned[year] = float(module().detach())
        print(f"    {tag} {year}: c={learned[year]:.3f} "
              f"({len(tr)} folds, {time.time()-t0:.0f}s)", flush=True)
    return learned


def gerber_fn(c_map_or_val):
    def fn(R, s_, d_):
        c = (c_map_or_val[d_.year] if isinstance(c_map_or_val, dict)
             else c_map_or_val)
        return backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, c, "gs2022"))(R, s_, d_)
    return fn


rows, dm_rows, th_rows = [], [], []
for uname, SYMS in UNIVERSES.items():
    print(f"===== {uname}: {' '.join(SYMS)} =====", flush=True)
    def uni(d, SYMS=SYMS):
        return SYMS

    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets[SYMS].iloc[i - WINDOW:i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[SYMS].iloc[i:i + STEP].fillna(0.0).values),
        })

    # ambient correlation, for the mechanism map
    corrs = []
    for y in (2018, 2021, 2024):
        d0 = min(x for x in dates if x.year == y)
        i = dates.get_loc(d0)
        C = np.corrcoef(rets[SYMS].iloc[max(0, i - WINDOW):i].fillna(0.0).values,
                        rowvar=False)
        corrs.append(np.mean(C[np.triu_indices_from(C, 1)]))
    mean_corr = float(np.mean(corrs))
    print(f"  mean pairwise correlation: {mean_corr:.3f}", flush=True)

    learned_c = train_tglobal(folds, uname)
    for y, c in sorted(learned_c.items()):
        th_rows.append({"universe": uname, "year": y, "c": round(c, 4),
                        "mean_corr": round(mean_corr, 4)})

    eval_start = min(f["date"] for f in folds if f["date"].year >= 2017)
    contenders = {"tglobal": gerber_fn(learned_c),
                  "gerber_c0.5": gerber_fn(0.5)}
    for c in GRID:
        contenders[f"grid_c{c}"] = gerber_fn(c)
    contenders.update({
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
        "equal_weight": lambda R, s, d: np.full(R.shape[1], 1.0 / R.shape[1]),
        "sample": backtest.covariance_weight_fn(baselines.sample_cov),
    })

    runs = {}
    for name, wfn in contenders.items():
        runs[name] = backtest.run_walkforward(name, rets, wfn, uni, eval_start,
                                              dates[-1], window=WINDOW,
                                              step=STEP)
        m = runs[name].metrics(cost_bps=10)
        rows.append({"universe": uname, "method": name,
                     "ann_ret_net": round(float(m["ann_ret_net"]), 5),
                     "ann_vol_net": round(float(m["ann_vol_net"]), 5),
                     "sharpe_net": round(float(m["sharpe_net"]), 5),
                     "sortino_net": round(float(m["sortino_net"]), 5),
                     "max_drawdown": round(float(m["max_drawdown"]), 5),
                     "avg_turnover": round(float(m["avg_turnover"]), 5)})
        if not name.startswith("grid_"):
            print(f"  {name:12s} ret={m['ann_ret_net']*100:6.2f}%  "
                  f"vol={m['ann_vol_net']*100:6.2f}%  SR={m['sharpe_net']:+.3f}  "
                  f"mdd={m['max_drawdown']*100:6.1f}%", flush=True)

    for a, b in [("tglobal", "gerber_c0.5"), ("tglobal", "ledoit_wolf"),
                 ("tglobal", "ans"), ("tglobal", "equal_weight"),
                 ("gerber_c0.5", "equal_weight")]:
        dm, p = stats.diebold_mariano(runs[a].fold_losses(),
                                      runs[b].fold_losses())
        dm_rows.append({"universe": uname, "method": a, "vs": b,
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})
        print(f"  DM {a} vs {b}: {dm:+.2f} (p={p:.4f})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "multiasset_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "multiasset_dm.csv", index=False)
pd.DataFrame(th_rows).to_csv(RESULTS / "multiasset_thresholds.csv", index=False)

print("\n=== ceiling first: what can a constant threshold add here? ===")
for uname in UNIVERSES:
    s = df[df.universe == uname].set_index("method")
    g = s[s.index.str.startswith("grid_")]
    d0, lr = s.loc["gerber_c0.5"], s.loc["tglobal"]
    orc = g.loc[g.sharpe_net.idxmax()]
    print(f"{uname}: default SR={d0.sharpe_net:.3f} vol={d0.ann_vol_net*100:.2f} | "
          f"learned SR={lr.sharpe_net:.3f} vol={lr.ann_vol_net*100:.2f} | "
          f"oracle {orc.name.replace('grid_c','c=')} SR={orc.sharpe_net:.3f} "
          f"-> headroom {orc.sharpe_net-d0.sharpe_net:+.3f}, "
          f"captured {lr.sharpe_net-d0.sharpe_net:+.3f}")
    ew = s.loc["equal_weight"]
    print(f"      vs EW: EW SR={ew.sharpe_net:.3f} vol={ew.ann_vol_net*100:.2f} | "
          f"family beats EW on Sharpe: "
          f"{sum(s.loc[m].sharpe_net > ew.sharpe_net for m in ('tglobal','gerber_c0.5','ledoit_wolf','ans','hrp'))}/5")
print("\nmulti-asset experiment done.")
