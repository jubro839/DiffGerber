"""Headline data pack: sector-balanced universes at N = 30, 50, 70.

The paper's primary evidence now comes from these three universes, so all
three need results at the same level of detail. N=70 was added in a later
round and lacks the learned shrinkage intensity, sub-period decomposition,
cost curve and certainty equivalents that N=30 and N=50 already have; this
script fills those gaps and re-emits all three in one consistent format.

Everything follows the frozen protocol: window 252 / step 21, evaluation
2017+, annual expanding refit with a horizon-aware boundary, warm-started
module with a fresh Adam at lr 0.02 for 40 epochs, tau annealed 0.2 -> 0.02,
straight-through estimator, gs2022 normalization, scale-relative ridge 1e-3,
simple-return metrics. Thresholds already fitted under that protocol are
reused; only the missing ones are trained.

Outputs (all keyed by N):
  headline_metrics.csv    five-metric panel at 0/10/25/50 bp, every arm
  headline_dm.csv         Diebold-Mariano of each arm against each baseline
  headline_subperiods.csv COVID crash, 2020, 2022, calm 2023-26
  headline_cer.csv        certainty equivalents at gamma in {2,5,10}
  headline_wealth.csv     net cumulative wealth paths
  headline_thresholds.csv learned threshold paths per arm
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
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,70").split(",")]
COSTS = [0.0, 10.0, 25.0, 50.0]
GAMMAS = [2.0, 5.0, 10.0]
WINDOWS = {"covid_crash": ("2020-02-19", "2020-03-23"),
           "year_2020": ("2020-01-01", "2020-12-31"),
           "year_2022": ("2022-01-01", "2022-12-31"),
           "calm_2023_26": ("2023-01-01", "2026-12-31")}
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]
BASELINES = ["gerber_c0.5", "ledoit_wolf", "ans", "hrp", "equal_weight"]

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


class ShrinkGerber(nn.Module):
    """Learned threshold and learned shrinkage intensity toward the sample."""

    def __init__(self):
        super().__init__()
        self.thr = GlobalThreshold(0.5)
        self.raw_d = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        return self.thr(), torch.sigmoid(self.raw_d)


def sample_cov_t(Rw):
    Rc = Rw - Rw.mean(0, keepdim=True)
    return Rc.T @ Rc / (Rw.shape[0] - 1)


def train_shrink(folds):
    torch.manual_seed(0)
    module = ShrinkGerber()
    layer = SoftGerber("gs2022", straight_through=True)
    per_year = {}
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
            for f in tr:
                c, delta = module()
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, c, tau)
                Sig = delta * (s.unsqueeze(-1) * G * s.unsqueeze(-2)) \
                    + (1 - delta) * sample_cov_t(f["Rw"])
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(tr)).backward()
            opt.step()
        with torch.no_grad():
            c, d_ = module()
            per_year[year] = (float(c), float(d_))
        print(f"    X2 {year}: c={per_year[year][0]:.4f} "
              f"delta={per_year[year][1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return per_year


def gerber_G(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


def window_metrics(net_log):
    simple = np.expm1(net_log)
    ann_ret = float(simple.mean() * 252)
    ann_vol = float(simple.std(ddof=1) * np.sqrt(252))
    dn = float(np.sqrt(np.mean(np.minimum(simple, 0.0) ** 2)) * np.sqrt(252))
    curve = np.exp(net_log.cumsum())
    return {"total_ret": float(np.expm1(net_log.sum())), "ann_ret": ann_ret,
            "ann_vol": ann_vol, "sharpe": ann_ret / ann_vol if ann_vol else np.nan,
            "sortino": ann_ret / dn if dn else np.nan,
            "mdd": float((curve / curve.cummax() - 1).min())}


rows, dm_rows, sub_rows, cer_rows, th_rows, wealth = [], [], [], [], [], []

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
    if len(xp):
        x2 = {int(r.year): (float(r.c_up), float(r.delta)) for r in xp.itertuples()}
        print("  reusing saved shrinkage parameters", flush=True)
    else:
        print("  training shrinkage intensity (not previously fitted)", flush=True)
        folds = []
        for i in range(WINDOW, len(dates) - 1, STEP):
            syms = uni(dates[i])
            folds.append({
                "date": dates[i],
                "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values),
                "Rf": torch.as_tensor(rets[syms].iloc[i:i + STEP].fillna(0.0).values)})
        x2 = train_shrink(folds)

    for y, (cu, cd) in cmap["tglobal"].items():
        th_rows.append({"N": N, "arm": "DG-GMV", "year": y, "c_up": round(cu, 4),
                        "c_down": None, "delta": None})
    for y, (cu, cd) in cmap["tasym"].items():
        th_rows.append({"N": N, "arm": "DG-Asym", "year": y, "c_up": round(cu, 4),
                        "c_down": None if cd is None else round(cd, 4), "delta": None})
    for y, (c, d_) in x2.items():
        th_rows.append({"N": N, "arm": "DG-Shrink", "year": y, "c_up": round(c, 4),
                        "c_down": None, "delta": round(d_, 4)})

    def gmv_fn(mod):
        def fn(R, symbols, as_of):
            cu, cd = cmap[mod][as_of.year]
            G, s = gerber_G(R, cu, cd)
            return backtest.gmv_from_cov(np.outer(s, s) * G)
        return fn

    def hrp_g(R, symbols, as_of):
        cu, _ = cmap["tglobal"][as_of.year]
        G, _s = gerber_G(R, cu)
        return np.asarray(baselines.hrp_weights(R, corr=G))

    def x2_fn(R, symbols, as_of):
        c, delta = x2.get(as_of.year, (0.5, 0.5))
        G, s = gerber_G(R, c)
        S = np.cov(R, rowvar=False, ddof=1)
        return backtest.gmv_from_cov(delta * np.outer(s, s) * G + (1 - delta) * S)

    arms = {"DG-GMV": gmv_fn("tglobal"), "DG-Asym": gmv_fn("tasym"),
            "DG-HRP": hrp_g, "DG-Shrink": x2_fn}
    peers = {
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
        "equal_weight": lambda R, s, d: np.full(R.shape[1], 1.0 / R.shape[1]),
    }

    runs = {}
    for name, fn in {**arms, **peers}.items():
        r = backtest.run_walkforward(name, rets, fn, uni, report_start, run_end,
                                     window=WINDOW, step=STEP)
        runs[name] = r
        for cb in COSTS:
            m = r.metrics(cost_bps=cb)
            rows.append({"N": N, "method": name, "cost_bps": cb,
                         **{k: round(float(m[k]), 5) for k in METRIC_KEYS}})
        r_log = r.daily_returns.copy()
        cost = pd.Series(0.0, index=r_log.index)
        for f in r.folds:
            if f.date in cost.index:
                cost.loc[f.date] += f.turnover * 10.0 * 1e-4
        net_log = r_log - cost
        wealth.append(np.exp(net_log.cumsum()).rename(f"N{N}|{name}"))
        for wname, (w0, w1) in WINDOWS.items():
            seg = net_log.loc[w0:w1]
            if len(seg) >= 5:
                sub_rows.append({"N": N, "method": name, "window": wname,
                                 **{k: round(v, 5)
                                    for k, v in window_metrics(seg).items()}})
        m10 = r.metrics(cost_bps=10.0)
        for gm in GAMMAS:
            cer_rows.append({"N": N, "method": name, "gamma": gm,
                             "cer": round(float(m10["ann_ret_net"]
                                                - gm / 2 * m10["ann_vol_net"] ** 2), 5)})
        print(f"  {name:14s} ret={m10['ann_ret_net']*100:6.2f}%  "
              f"vol={m10['ann_vol_net']*100:6.2f}%  SR={m10['sharpe_net']:+.2f}  "
              f"So={m10['sortino_net']:+.2f}  mdd={m10['max_drawdown']*100:6.1f}%",
              flush=True)

    for a in arms:
        for b in BASELINES:
            al, bl = runs[a].fold_losses(), runs[b].fold_losses()
            n_ = min(len(al), len(bl))
            dm, p = stats.diebold_mariano(al[:n_], bl[:n_])
            dm_rows.append({"N": N, "method": a, "vs": b,
                            "dm": round(float(dm), 3), "p": round(float(p), 5)})

pd.DataFrame(rows).to_csv(RESULTS / "headline_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "headline_dm.csv", index=False)
pd.DataFrame(sub_rows).to_csv(RESULTS / "headline_subperiods.csv", index=False)
pd.DataFrame(cer_rows).to_csv(RESULTS / "headline_cer.csv", index=False)
pd.DataFrame(th_rows).to_csv(RESULTS / "headline_thresholds.csv", index=False)
pd.concat(wealth, axis=1).to_csv(RESULTS / "headline_wealth.csv")
print("\nheadline pack done.")
