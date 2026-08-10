"""Mixed universes: sector-balanced stocks PLUS a multi-asset ETF sleeve.

One portfolio holds both: the point-in-time sector-balanced equity selection
(N in {30, 50, 70}, as in the headline design) and a fixed nine-ETF sleeve
spanning bonds, credit, TIPS, gold, and real estate --

    sleeve: IWM EFA EEM AGG HYG TIP GLD VNQ TLT

(SPY is excluded from the sleeve because its constituents overlap the equity
selection; IWM/EFA/EEM add small-cap and international equity exposure that
the S&P names do not carry.) The estimator therefore faces equity-equity,
equity-bond, and bond-real-asset pairs inside a single covariance matrix,
with volatilities ranging from ~4% (AGG) to ~25% (single stocks).

Protocol is frozen as everywhere else: window 252 / step 21, evaluation
2017+, 10bp costs, annual expanding refit with warm start, Adam 0.02 x 40
epochs, tau 0.2 -> 0.02, gs2022, scale-relative ridge 1e-3. Ceiling first:
the constant-threshold grid is measured alongside the learner.

NOTE the sample cannot be extended before 2015: the licensed price warehouse
begins 2015-01-02 (membership history goes back decades, prices do not).

Outputs: mixed_metrics.csv, mixed_dm.csv, mixed_thresholds.csv
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
from diffgerber.soft_gerber import SoftGerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP, EPOCHS = 252, 21, 40
SIZES = [30, 50, 70]
SLEEVE = ["IWM", "EFA", "EEM", "AGG", "HYG", "TIP", "GLD", "VNQ", "TLT"]
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
eq_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(set(eq_syms) | set(SLEEVE)))
rets = data.log_returns(px)
dates = rets.index
fold_dates = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)]
report_start = min(d for d in fold_dates if d.year >= 2017)
run_end = max(fold_dates) + pd.Timedelta(days=1)


def gerber_fn(c_map_or_val):
    def fn(R, s_, d_):
        c = (c_map_or_val[d_.year] if isinstance(c_map_or_val, dict)
             else c_map_or_val)
        return backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, c, "gs2022"))(R, s_, d_)
    return fn


def train_tglobal(folds, tag):
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


rows, dm_rows, th_rows = [], [], []
for N in SIZES:
    uname = f"mix{N}"
    print(f"===== {uname}: sector-{N} stocks + {len(SLEEVE)}-ETF sleeve "
          f"({N + len(SLEEVE)} assets) =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = list(data.pit_universe_sector(d, N, px, mem)) + SLEEVE
        return _m[d]

    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i:i + STEP].fillna(0.0).values),
        })

    corrs = []
    for y in (2018, 2021, 2024):
        d0 = min(x for x in dates if x.year == y)
        i = dates.get_loc(d0)
        syms = uni(dates[i])
        C = np.corrcoef(rets[syms].iloc[max(0, i - WINDOW):i].fillna(0.0).values,
                        rowvar=False)
        corrs.append(np.mean(C[np.triu_indices_from(C, 1)]))
    mean_corr = float(np.mean(corrs))
    print(f"  mean pairwise correlation: {mean_corr:.3f}", flush=True)

    learned_c = train_tglobal(folds, uname)
    for y, c in sorted(learned_c.items()):
        th_rows.append({"universe": uname, "year": y, "c": round(c, 4),
                        "mean_corr": round(mean_corr, 4)})

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
    })

    runs = {}
    for name, wfn in contenders.items():
        runs[name] = backtest.run_walkforward(name, rets, wfn, uni,
                                              report_start, run_end,
                                              window=WINDOW, step=STEP)
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
                  f"mdd={m['max_drawdown']*100:6.1f}%  "
                  f"to={m['avg_turnover']:.2f}", flush=True)

    for a, b in [("tglobal", "gerber_c0.5"), ("tglobal", "ledoit_wolf"),
                 ("tglobal", "ans"), ("tglobal", "hrp"),
                 ("tglobal", "equal_weight"),
                 ("gerber_c0.5", "equal_weight")]:
        dm, p = stats.diebold_mariano(runs[a].fold_losses(),
                                      runs[b].fold_losses())
        dm_rows.append({"universe": uname, "method": a, "vs": b,
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})
        print(f"  DM {a} vs {b}: {dm:+.2f} (p={p:.4f})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "mixed_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "mixed_dm.csv", index=False)
pd.DataFrame(th_rows).to_csv(RESULTS / "mixed_thresholds.csv", index=False)

print("\n=== ceiling: what can a constant threshold add in the mix? ===")
for N in SIZES:
    uname = f"mix{N}"
    s = df[df.universe == uname].set_index("method")
    g = s[s.index.str.startswith("grid_")]
    d0, lr = s.loc["gerber_c0.5"], s.loc["tglobal"]
    ov = g.loc[g.ann_vol_net.idxmin()]
    os_ = g.loc[g.sharpe_net.idxmax()]
    print(f"{uname}: vol  default {d0.ann_vol_net*100:.2f} | learned "
          f"{lr.ann_vol_net*100:.2f} | grid-best {ov.ann_vol_net*100:.2f} "
          f"({ov.name.replace('grid_c','c=')})")
    print(f"        SR   default {d0.sharpe_net:.3f} | learned "
          f"{lr.sharpe_net:.3f} | grid-best {os_.sharpe_net:.3f} "
          f"({os_.name.replace('grid_c','c=')}) | EW "
          f"{s.loc['equal_weight','sharpe_net']:.3f} at "
          f"{s.loc['equal_weight','ann_vol_net']*100:.2f}% vol")
print("\nmixed-universe experiment done.")
