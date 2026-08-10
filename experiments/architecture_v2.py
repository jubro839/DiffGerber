"""DiffGerber-Ensemble: rolling architecture selection on sector universes.

Arm pool (all built on the learned-threshold Gerber matrix):
  A1 gmv_tglobal     GMV allocation, variance-trained threshold
  A2 gmv_tasym       GMV allocation, variance-trained asymmetric thresholds
  A3 hrp_gerber      HRP allocation on the Gerber correlation (A1's thresholds)
  A4 gmv_dd          GMV allocation, downside-trained threshold
  A5 hrp_gerber_dd   HRP allocation on the Gerber correlation (A4's thresholds)

Rolling rule (fixed in advance): at the start of each evaluation year Y, pick
the arm with the highest realized Sortino over the trailing three years of
folds (2016 only, for the first selection). Every fold used for the selection
ends strictly before Y. Ensemble performance is therefore fully out-of-sample.
Learned thresholds for 2016 folds are the untrained initial value c=0.5 for
every learned arm (no training data precedes them under the refit rule).

Baselines: fixed Gerber c=0.5 (GMV), Ledoit-Wolf, ANS, plain HRP.
Metrics: AnnRet, AnnVol, Sharpe, Sortino, MDD (+ turnover), net of 10 bps,
reported on 2017+. An in-sample ceiling row (best single arm chosen with
hindsight, per metric) is included for diagnosis and labelled as such.
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import (decision_loss, downside_decision_loss,
                                        gmv_closed_form)
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import AsymThreshold, GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,100").split(",")]
COST = 10.0
TRAIL_YEARS = 3
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

# hrp regression check: corr=None must equal explicit sample correlation
_R = np.random.default_rng(0).standard_normal((300, 12)) * 0.01
assert np.allclose(baselines.hrp_weights(_R),
                   baselines.hrp_weights(_R, corr=np.corrcoef(_R, rowvar=False)))

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index


def train_threshold(folds, kind, loss_fn):
    """Annual expanding refit under the frozen protocol; returns {year: (cu, cd)}.
    Year 2016 (pre-training) maps to the initial value 0.5."""
    torch.manual_seed(0)
    module = GlobalThreshold(0.5) if kind == "tglobal" else AsymThreshold(0.5)
    layer = SoftGerber("gs2022", straight_through=True)
    per_year = {2016: (0.5, None if kind == "tglobal" else 0.5)}
    for year in range(2017, LAST_YEAR + 1):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = [f for f in folds
                 if dates[min(dates.get_loc(f["date"]) + STEP,
                              len(dates) - 1)] < cutoff]
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in train:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                if kind == "tglobal":
                    G = layer(f["Rw"] / s, module(), tau)
                else:
                    cu, cd = module()
                    G = layer(f["Rw"] / s, cu, tau, c_down=cd)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + loss_fn(gmv_closed_form(Sig), f["Rf"])
            (loss / len(train)).backward()
            opt.step()
        with torch.no_grad():
            if kind == "tglobal":
                per_year[year] = (float(module()), None)
            else:
                cu, cd = module()
                per_year[year] = (float(cu), float(cd))
    return per_year


def gerber_corr(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


def slice_metrics(res, start, cost_bps=COST, periods=252):
    """Metrics on folds dated >= start (same conventions as backtest.metrics)."""
    keep = [f for f in res.folds if f.date >= start]
    r_log = res.daily_returns.loc[keep[0].date:]
    cost = pd.Series(0.0, index=r_log.index)
    for f in keep:
        if f.date in cost.index:
            cost.loc[f.date] += f.turnover * cost_bps * 1e-4
    net_log = r_log - cost
    net = np.expm1(net_log)
    vol = net.std(ddof=1) * np.sqrt(periods)
    ret = net.mean() * periods
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)) * np.sqrt(periods)
    curve = np.exp(net_log.cumsum())
    to = [f.turnover for f in keep]
    return {"ann_ret_net": ret, "ann_vol_net": vol,
            "sharpe_net": ret / vol if vol > 0 else np.nan,
            "sortino_net": ret / downside if downside > 0 else np.nan,
            "max_drawdown": float((curve / curve.cummax() - 1).min()),
            "avg_turnover": float(np.mean(to[1:]) if len(to) > 1 else to[0])}


def yearly_sortino(res, years, cost_bps=COST):
    """Realized Sortino over the folds of the given years (selection score)."""
    keep = [f for f in res.folds if f.date.year in years]
    if not keep:
        return -np.inf
    r_log = pd.concat([res.daily_returns.loc[f.date:].iloc[:STEP] for f in keep])
    r_log = r_log[~r_log.index.duplicated()]
    cost = sum(f.turnover for f in keep) * cost_bps * 1e-4 / max(len(r_log), 1)
    net = np.expm1(r_log - cost)
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2))
    return float(net.mean() / downside) if downside > 0 else -np.inf


METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]
rows, sel_rows, dm_rows = [], [], []

for N in SIZES:
    print(f"===== N={N} =====", flush=True)
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
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        })
    all_dates = [f["date"] for f in folds if f["date"].year <= LAST_YEAR]
    run_start, run_end = min(all_dates), max(all_dates) + pd.Timedelta(days=1)
    report_start = min(d for d in all_dates if d.year >= 2017)

    t0 = time.time()
    c_var = train_threshold(folds, "tglobal", decision_loss)
    c_asym = train_threshold(folds, "tasym", decision_loss)
    c_dd = train_threshold(folds, "tglobal", downside_decision_loss)
    print(f"  thresholds trained ({time.time()-t0:.0f}s)  "
          f"var2026={c_var[LAST_YEAR][0]:.2f}  "
          f"asym2026={tuple(round(x,2) for x in c_asym[LAST_YEAR])}  "
          f"dd2026={c_dd[LAST_YEAR][0]:.2f}", flush=True)

    def gmv_fn(cmap):
        def fn(R, symbols, as_of):
            cu, cd = cmap[as_of.year]
            G, s = gerber_corr(R, cu, cd)
            return backtest.gmv_from_cov(np.outer(s, s) * G)
        return fn

    def hrp_fn(cmap):
        def fn(R, symbols, as_of):
            cu, cd = cmap[as_of.year]
            G, _ = gerber_corr(R, cu, cd)
            return baselines.hrp_weights(R, corr=G)
        return fn

    arms = {
        "A1_gmv_tglobal": gmv_fn(c_var),
        "A2_gmv_tasym": gmv_fn(c_asym),
        "A3_hrp_gerber": hrp_fn(c_var),
        "A4_gmv_dd": gmv_fn(c_dd),
        "A5_hrp_gerber_dd": hrp_fn(c_dd),
    }
    baselines_fns = {
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
    }

    runs = {}
    for name, wfn in {**arms, **baselines_fns}.items():
        runs[name] = backtest.run_walkforward(name, rets, wfn, uni, run_start,
                                              run_end, window=WINDOW, step=STEP)
        m = slice_metrics(runs[name], report_start)
        rows.append({"N": N, "method": name,
                     **{k: round(float(m[k]), 5) for k in METRIC_KEYS}})
        print(f"  N={N} {name:16s} ret={m['ann_ret_net']*100:6.2f}%  "
              f"vol={m['ann_vol_net']*100:6.2f}%  SR={m['sharpe_net']:+.2f}  "
              f"So={m['sortino_net']:+.2f}  mdd={m['max_drawdown']*100:6.1f}%",
              flush=True)

    # ---- rolling selection (look-ahead-free by construction) ----
    ens_folds, choices = [], {}
    for year in range(2017, LAST_YEAR + 1):
        trail = [y for y in range(year - TRAIL_YEARS, year) if y >= 2016]
        scores = {a: yearly_sortino(runs[a], trail) for a in arms}
        pick = max(scores, key=lambda a: (scores[a],
                                          -slice_metrics(runs[a], report_start)
                                          ["ann_vol_net"]))
        choices[year] = pick
        for f in runs[pick].folds:
            if f.date.year == year:
                assert all(t < year for t in trail)      # no look-ahead
                ens_folds.append((pick, f))
        sel_rows.append({"N": N, "year": year, "picked": pick,
                         **{a: round(scores[a], 4) for a in arms}})
    # assemble ensemble daily series with each arm's own costs + switch cost
    seg, prev_arm = [], None
    ens_fold_objs = []
    for arm, f in ens_folds:
        r_log = runs[arm].daily_returns.loc[f.date:].iloc[:STEP].copy()
        extra = f.turnover
        if prev_arm is not None and arm != prev_arm and f is ens_folds[0][1]:
            pass
        if prev_arm is not None and arm != prev_arm:
            extra += 2.0                                  # conservative switch cost
        r_log.iloc[0] -= extra * COST * 1e-4
        seg.append(r_log)
        prev_arm = arm
        ens_fold_objs.append(f)
    ens_log = pd.concat(seg)
    ens_log = ens_log[~ens_log.index.duplicated()]
    net = np.expm1(ens_log)
    vol = net.std(ddof=1) * np.sqrt(252)
    ret = net.mean() * 252
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)) * np.sqrt(252)
    curve = np.exp(ens_log.cumsum())
    m_ens = {"ann_ret_net": ret, "ann_vol_net": vol,
             "sharpe_net": ret / vol, "sortino_net": ret / downside,
             "max_drawdown": float((curve / curve.cummax() - 1).min()),
             "avg_turnover": float(np.mean([f.turnover for _, f in ens_folds]))}
    rows.append({"N": N, "method": "ENSEMBLE_rolling",
                 **{k: round(float(m_ens[k]), 5) for k in METRIC_KEYS}})
    print(f"  N={N} {'ENSEMBLE':16s} ret={ret*100:6.2f}%  vol={vol*100:6.2f}%  "
          f"SR={ret/vol:+.2f}  So={ret/downside:+.2f}  "
          f"mdd={m_ens['max_drawdown']*100:6.1f}%  picks={choices}", flush=True)

    # in-sample ceiling (labelled): best arm per metric with hindsight
    ceil = {}
    for k in ["ann_ret_net", "sharpe_net", "sortino_net"]:
        ceil[k] = max((slice_metrics(runs[a], report_start)[k], a) for a in arms)
    for k in ["ann_vol_net", "max_drawdown"]:
        best = min((slice_metrics(runs[a], report_start)[k], a) for a in arms) \
            if k == "ann_vol_net" else \
            max((slice_metrics(runs[a], report_start)[k], a) for a in arms)
        ceil[k] = best
    rows.append({"N": N, "method": "CEILING_insample(REF_ONLY)",
                 **{k: round(float(ceil.get(k, (np.nan,))[0]), 5)
                    if k in ceil else np.nan for k in METRIC_KEYS}})

    ens_losses = np.array([f.realized_var for _, f in ens_folds])
    for b in ["hrp", "gerber_c0.5", "ans"]:
        bl = np.array([f.realized_var for f in runs[b].folds
                       if f.date >= report_start])
        n_ = min(len(ens_losses), len(bl))
        dm, p = stats.diebold_mariano(ens_losses[:n_], bl[:n_])
        dm_rows.append({"N": N, "vs": b, "dm": round(float(dm), 3),
                        "p": round(float(p), 5)})
        print(f"    ENSEMBLE vs {b}: DM={dm:+.2f} p={p:.4f}", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "arch_v2_metrics.csv", index=False)
pd.DataFrame(sel_rows).to_csv(RESULTS / "arch_v2_selection.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "arch_v2_dm.csv", index=False)
print("\narchitecture v2 done.")
