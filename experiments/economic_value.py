"""Tier-1 economic-value report: no new selection decisions, no retraining.

Re-runs the already-selected portfolios (thresholds loaded from
arch_v2b_thresholds_N{N}.csv) and the four baselines on the sector-balanced
universes, then re-expresses the SAME portfolios along four standard axes:

  1. Transaction-cost sensitivity at {0, 10, 25, 50} bps
     (DeMiguel-Garlappi-Uppal 2009 stress up to 50 bps).
  2. Certainty-equivalent return, CER = mu - gamma/2 * sigma^2, gamma in
     {2, 5, 10} (Fleming-Kirby-Ostdiek 2001 economic-value framing).
  3. Crisis sub-periods: COVID crash (2020-02-19..2020-03-23), 2020, 2022.
  4. One-sided DM vs each baseline + intersection-union test (a uniform
     "beats ALL baselines" claim holds at level alpha iff the MAX one-sided
     p across baselines < alpha; Berger 1982) + Hansen (2005) SPA-style
     bootstrap check that no baseline significantly outperforms us.

Everything is a deterministic re-expression of portfolios fixed by earlier
pre-specified runs; nothing here feeds back into model choice.
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
GAMMAS = [2.0, 5.0, 10.0]
CRISIS = {
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "year_2020": ("2020-01-01", "2020-12-31"),
    "year_2022": ("2022-01-01", "2022-12-31"),
    "calm_2023_26": ("2023-01-01", "2026-12-31"),
}
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index

OURS = ["A1_gmv_tglobal", "A2_gmv_tasym", "A3_hrp_gerber", "BLEND_50_50"]
BASE = ["gerber_c0.5", "ledoit_wolf", "ans", "hrp"]


def gerber_G(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


def net_log_series(run, cost_bps):
    r_log = run.daily_returns.copy()
    cost = pd.Series(0.0, index=r_log.index)
    for f in run.folds:
        if f.date in cost.index:
            cost.loc[f.date] += f.turnover * cost_bps * 1e-4
    return r_log - cost


def spa_pvalue(d, n_boot=5000, mean_block=6, seed=0):
    """Hansen (2005)-style SPA bootstrap. d: (n_folds, n_baselines) with
    d[t, b] = loss_ours[t] - loss_base[b, t]; positive mean(d[:, b]) means
    baseline b beats us on the decision loss. H0: no baseline outperforms.
    Stationary bootstrap over fold index (joint across baselines)."""
    rng = np.random.default_rng(seed)
    n, m = d.shape
    mu = d.mean(0)
    omega = d.std(0, ddof=1)
    omega = np.where(omega > 0, omega, 1e-12)
    t_stat = np.max(np.sqrt(n) * mu / omega)
    # Hansen's recentering: keep means that are not "too negative"
    keep = mu >= -omega * np.sqrt(2.0 * np.log(np.log(max(n, 3))) / n)
    center = np.where(keep, mu, 0.0)
    p_geom = 1.0 / mean_block
    count = 0
    for _ in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p_geom else (idx[t - 1] + 1) % n
        db = d[idx]
        mub = db.mean(0)
        t_b = np.max(np.sqrt(n) * (mub - center) / omega)
        if t_b >= t_stat:
            count += 1
    return t_stat, count / n_boot


cost_rows, cer_rows, crisis_rows, test_rows = [], [], [], []

for N in SIZES:
    print(f"===== N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    th = pd.read_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv")
    cmaps = {}
    for mod in ["tglobal", "tasym", "tglobal_dd"]:
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

    runs = {}
    for name, wfn in contenders.items():
        r = backtest.run_walkforward(name, rets, wfn, uni, report_start,
                                     run_end, window=WINDOW, step=STEP)
        runs[name] = r
        for cb in COSTS:
            mm = r.metrics(cost_bps=cb)
            cost_rows.append({"N": N, "method": name, "cost_bps": cb,
                              **{k: round(float(mm[k]), 5)
                                 for k in ["ann_ret_net", "ann_vol_net",
                                           "sharpe_net", "sortino_net",
                                           "max_drawdown", "avg_turnover"]}})
        mm = r.metrics(cost_bps=10.0)
        for g in GAMMAS:
            cer = mm["ann_ret_net"] - g / 2.0 * mm["ann_vol_net"] ** 2
            cer_rows.append({"N": N, "method": name, "gamma": g,
                             "cer": round(float(cer), 5)})
        net_log = net_log_series(r, 10.0)
        for wname, (w0, w1) in CRISIS.items():
            seg = net_log.loc[w0:w1]
            if len(seg) < 5:
                continue
            simple = np.expm1(seg)
            curve = np.exp(seg.cumsum())
            crisis_rows.append({
                "N": N, "method": name, "window": wname,
                "total_ret": round(float(np.expm1(seg.sum())), 5),
                "ann_vol": round(float(simple.std(ddof=1) * np.sqrt(252)), 5),
                "mdd": round(float((curve / curve.cummax() - 1).min()), 5),
            })
        print(f"  {name:16s} done ({len(r.folds)} folds)", flush=True)

    for ours in OURS:
        a_l = runs[ours].fold_losses()
        pones = {}
        for b in BASE:
            b_l = runs[b].fold_losses()
            n_ = min(len(a_l), len(b_l))
            dm, p2 = stats.diebold_mariano(a_l[:n_], b_l[:n_])
            p1 = p2 / 2.0 if dm < 0 else 1.0 - p2 / 2.0
            pones[b] = p1
            test_rows.append({"N": N, "method": ours, "vs": b, "test": "dm",
                              "stat": round(float(dm), 3),
                              "p_two": round(float(p2), 5),
                              "p_one": round(float(p1), 5)})
        test_rows.append({"N": N, "method": ours, "vs": "ALL", "test": "iut",
                          "stat": None, "p_two": None,
                          "p_one": round(float(max(pones.values())), 5)})
        d = np.column_stack([a_l[:min(len(a_l), len(runs[b].fold_losses()))]
                             - runs[b].fold_losses()[:min(len(a_l),
                                                          len(runs[b].fold_losses()))]
                             for b in BASE])
        t_spa, p_spa = spa_pvalue(d)
        test_rows.append({"N": N, "method": ours, "vs": "ALL", "test": "spa",
                          "stat": round(float(t_spa), 3), "p_two": None,
                          "p_one": round(float(p_spa), 5)})
        print(f"  tests {ours}: IUT p={max(pones.values()):.4f} "
              f"SPA p={p_spa:.4f}", flush=True)

pd.DataFrame(cost_rows).to_csv(RESULTS / "economic_value_cost_curve.csv", index=False)
pd.DataFrame(cer_rows).to_csv(RESULTS / "economic_value_cer.csv", index=False)
pd.DataFrame(crisis_rows).to_csv(RESULTS / "economic_value_crisis.csv", index=False)
pd.DataFrame(test_rows).to_csv(RESULTS / "economic_value_tests.csv", index=False)
print("\neconomic-value report done.")
