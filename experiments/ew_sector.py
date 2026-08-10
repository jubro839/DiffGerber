"""Equal weight on the sector-balanced universes.

The 1/N benchmark was added to the market-cap universes first; a reader will
immediately ask how it does on the sector-balanced ones, and so should we
before deciding which universe leads the paper. Same frozen protocol and the
same saved thresholds as architecture_v2b, so the learned arms here are the
portfolios already reported; only the 1/N row and the tests against it are
new.
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
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,100").split(",")]
COSTS = [0.0, 10.0, 25.0, 50.0]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index


def gerber_G(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


rows, dm_rows, wealth = [], [], []

for N in SIZES:
    print(f"===== sector N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    th = pd.read_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv")
    cmap = {}
    for mod in ("tglobal", "tasym"):
        sub = th[th.module == mod]
        cmap[mod] = {int(r.year): (float(r.c_up),
                                   None if pd.isna(r.c_down) else float(r.c_down))
                     for r in sub.itertuples()}
    xp = pd.read_csv(RESULTS / "prereg_extensions_params.csv")
    xp = xp[(xp.N == N) & (xp.module == "X2_shrink")]
    x2 = {int(r.year): (float(r.c_up), float(r.delta)) for r in xp.itertuples()}

    def gmv_fn(mod):
        def fn(R, symbols, as_of):
            cu, cd = cmap[mod][as_of.year]
            G, s = gerber_G(R, cu, cd)
            return backtest.gmv_from_cov(np.outer(s, s) * G)
        return fn

    def hrp_fn(R, symbols, as_of):
        cu, _ = cmap["tglobal"][as_of.year]
        G, _s = gerber_G(R, cu)
        return np.asarray(baselines.hrp_weights(R, corr=G))

    def x2_fn(R, symbols, as_of):
        c, delta = x2.get(as_of.year, (0.5, 0.5))
        G, s = gerber_G(R, c)
        S = np.cov(R, rowvar=False, ddof=1)
        return backtest.gmv_from_cov(delta * np.outer(s, s) * G + (1 - delta) * S)

    contenders = {
        "DG-GMV": gmv_fn("tglobal"),
        "DG-Asym": gmv_fn("tasym"),
        "DG-HRP": hrp_fn,
        "DG-Shrink": x2_fn,
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
        "equal_weight": lambda R, s, d: np.full(R.shape[1], 1.0 / R.shape[1]),
    }

    fold_dates = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)
                  if dates[i].year <= LAST_YEAR]
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
        r_log = r.daily_returns.copy()
        cost = pd.Series(0.0, index=r_log.index)
        for f in r.folds:
            if f.date in cost.index:
                cost.loc[f.date] += f.turnover * 10.0 * 1e-4
        wealth.append(np.exp((r_log - cost).cumsum()).rename(f"N{N}|{name}"))
        mm = r.metrics(cost_bps=10.0)
        print(f"  {name:14s} ret={mm['ann_ret_net']*100:6.2f}%  "
              f"vol={mm['ann_vol_net']*100:6.2f}%  SR={mm['sharpe_net']:+.2f}  "
              f"So={mm['sortino_net']:+.2f}  mdd={mm['max_drawdown']*100:6.1f}%",
              flush=True)

    for ours in ["DG-GMV", "DG-Asym", "DG-HRP", "DG-Shrink"]:
        a_l, b_l = runs[ours].fold_losses(), runs["equal_weight"].fold_losses()
        n_ = min(len(a_l), len(b_l))
        dm, p = stats.diebold_mariano(a_l[:n_], b_l[:n_])
        dm_rows.append({"N": N, "method": ours, "vs": "equal_weight",
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})
        print(f"    {ours} vs 1/N: DM={dm:+.2f} p={p:.4f}", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "ew_sector_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "ew_sector_dm.csv", index=False)
pd.concat(wealth, axis=1).to_csv(RESULTS / "wealth_paths_sector.csv")
print("\nsector equal-weight comparison done.")
