"""Full 5-metric table per sub-period window: extends the Tier-1 crisis
decomposition (economic_value.py) with Sharpe / Sortino / annualized return
per window. Same portfolios (thresholds from arch_v2b_thresholds_N{N}.csv),
same windows, 10 bps costs — a re-expression, no new decisions.

Note: the covid_crash window is 23 trading days; annualized ratios on it are
reported for completeness but are dominated by the window's sign.
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data
from diffgerber.soft_gerber import hard_gerber

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,100").split(",")]
WINDOWS = {
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "year_2020": ("2020-01-01", "2020-12-31"),
    "year_2022": ("2022-01-01", "2022-12-31"),
    "calm_2023_26": ("2023-01-01", "2026-12-31"),
    "full_2017_26": ("2017-01-01", "2026-12-31"),
}
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

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


def window_metrics(net_log):
    simple = np.expm1(net_log)
    ann_ret = simple.mean() * 252
    ann_vol = simple.std(ddof=1) * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(simple, 0.0) ** 2)) * np.sqrt(252)
    curve = np.exp(net_log.cumsum())
    return {
        "total_ret": float(np.expm1(net_log.sum())),
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "sortino": float(ann_ret / downside) if downside > 0 else np.nan,
        "mdd": float((curve / curve.cummax() - 1).min()),
    }


rows = []
for N in SIZES:
    print(f"===== N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    th = pd.read_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv")
    cmaps = {}
    for mod in ["tglobal", "tasym"]:
        sub = th[th.module == mod]
        cmaps[mod] = {int(r.year): (float(r.c_up),
                                    None if pd.isna(r.c_down) else float(r.c_down))
                      for r in sub.itertuples()}

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

    contenders = {
        "A1_gmv_tglobal": a1,
        "A2_gmv_tasym": gmv_fn("tasym"),
        "A3_hrp_gerber": a3,
        "BLEND_50_50": blend,
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
    }

    fold_dates = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)
                  if dates[i].year <= LAST_YEAR]
    report_start = min(d for d in fold_dates if d.year >= 2017)
    run_end = max(fold_dates) + pd.Timedelta(days=1)

    for name, wfn in contenders.items():
        r = backtest.run_walkforward(name, rets, wfn, uni, report_start,
                                     run_end, window=WINDOW, step=STEP)
        r_log = r.daily_returns.copy()
        cost = pd.Series(0.0, index=r_log.index)
        for f in r.folds:
            if f.date in cost.index:
                cost.loc[f.date] += f.turnover * 10.0 * 1e-4
        net_log = r_log - cost
        for wname, (w0, w1) in WINDOWS.items():
            seg = net_log.loc[w0:w1]
            if len(seg) < 5:
                continue
            rows.append({"N": N, "method": name, "window": wname,
                         **{k: round(v, 5)
                            for k, v in window_metrics(seg).items()}})
        print(f"  {name:16s} done", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "subperiod_metrics.csv", index=False)
print("\nsubperiod full-metric table done.")
