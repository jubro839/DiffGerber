"""Sector-balanced universes across N, with the S&P 500 index as benchmark.

Extends the sector sweep to N in {20, 40, 70} -- training their thresholds
under the frozen protocol, since only 30/50/100 had been fit -- and adds the
S&P 500 index as a market row, identical in every panel.

The index used is ^GSPC, which is a PRICE index: its return excludes
dividends and therefore sits roughly 1.5-2 percentage points per year below a
total-return benchmark such as SPY. Volatility and drawdown are essentially
unaffected. The row is labelled accordingly so the return column is not read
as a total-return comparison.
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import AsymThreshold, GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "20,40,70").split(",")]
COSTS = [0.0, 10.0, 25.0, 50.0]
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]

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


def gerber_G(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


def train_thresholds(uni, N):
    """Fit T-global and T-asym under the frozen protocol: annual expanding
    refit with a horizon-aware boundary, warm-started module, fresh Adam at
    lr 0.02 for 40 epochs, tau annealed 0.2 -> 0.02, straight-through."""
    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i:i + STEP].fillna(0.0).values),
        })
    out, saved = {}, []
    for kind in ("tglobal", "tasym"):
        torch.manual_seed(0)
        module = GlobalThreshold(0.5) if kind == "tglobal" else AsymThreshold(0.5)
        layer = SoftGerber("gs2022", straight_through=True)
        per_year = {}
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
                    loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
                (loss / len(train)).backward()
                opt.step()
            with torch.no_grad():
                if kind == "tglobal":
                    per_year[year] = (float(module()), None)
                else:
                    cu, cd = module()
                    per_year[year] = (float(cu), float(cd))
            saved.append({"N": N, "module": kind, "year": year,
                          "c_up": round(per_year[year][0], 4),
                          "c_down": (None if per_year[year][1] is None
                                     else round(per_year[year][1], 4))})
        out[kind] = per_year
        last = per_year[LAST_YEAR]
        print(f"  trained {kind}: c_2026={last[0]:.4f}"
              + ("" if last[1] is None else f" / {last[1]:.4f}"), flush=True)
    pd.DataFrame(saved).to_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv",
                               index=False)
    return out


def year_thresholds(N):
    th = pd.read_csv(RESULTS / f"arch_v2b_thresholds_N{N}.csv")
    out = {}
    for mod in ("tglobal", "tasym"):
        sub = th[th.module == mod]
        out[mod] = {int(r.year): (float(r.c_up),
                                  None if pd.isna(r.c_down) else float(r.c_down))
                    for r in sub.itertuples()}
    return out


# ---------------- market benchmark: S&P 500 index (price, ex-dividend) ----
rows, dm_rows, wealth = [], [], []

gspc = data.index_levels("^GSPC", "2015-01-01", "2026-07-28")
g = gspc["close_price"].astype(float)
g.index = pd.to_datetime(g.index)
g = g.loc[report_start:run_end]
gr = np.log(g).diff().dropna()
simple = np.expm1(gr)
ann_ret = float(simple.mean() * 252)
ann_vol = float(simple.std(ddof=1) * np.sqrt(252))
dn = float(np.sqrt(np.mean(np.minimum(simple, 0.0) ** 2)) * np.sqrt(252))
curve = np.exp(gr.cumsum())
mkt_mdd = float((curve / curve.cummax() - 1).min())
for cb in COSTS:                      # index is buy-and-hold: cost-invariant
    rows.append({"N": "market", "method": "sp500_index_price", "cost_bps": cb,
                 "ann_ret_net": round(ann_ret, 5),
                 "ann_vol_net": round(ann_vol, 5),
                 "sharpe_net": round(ann_ret / ann_vol, 5),
                 "sortino_net": round(ann_ret / dn, 5),
                 "max_drawdown": round(mkt_mdd, 5), "avg_turnover": 0.0})
wealth.append(curve.rename("market|sp500_index_price"))
print(f"  S&P 500 index (price, ex-div)  ret={ann_ret*100:6.2f}%  "
      f"vol={ann_vol*100:6.2f}%  SR={ann_ret/ann_vol:+.2f}  "
      f"So={ann_ret/dn:+.2f}  mdd={mkt_mdd*100:6.1f}%", flush=True)

# ---------------- sector universes ----------------------------------------
for N in SIZES:
    print(f"===== sector N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector(d, N, px, mem)
        return _m[d]

    if (RESULTS / f"arch_v2b_thresholds_N{N}.csv").exists():
        cmap = year_thresholds(N)
    else:
        print(f"  no saved thresholds for N={N}; training under the frozen "
              f"protocol", flush=True)
        cmap = train_thresholds(uni, N)

    contenders = {
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
        "equal_weight": lambda R, s, d: np.full(R.shape[1], 1.0 / R.shape[1]),
    }
    if cmap is not None:
        def gmv_fn(mod, cmap=cmap):
            def fn(R, symbols, as_of):
                cu, cd = cmap[mod][as_of.year]
                G, s = gerber_G(R, cu, cd)
                return backtest.gmv_from_cov(np.outer(s, s) * G)
            return fn

        def hrp_g(R, symbols, as_of, cmap=cmap):
            cu, _ = cmap["tglobal"][as_of.year]
            G, _s = gerber_G(R, cu)
            return np.asarray(baselines.hrp_weights(R, corr=G))

        contenders = {"DG-GMV": gmv_fn("tglobal"), "DG-Asym": gmv_fn("tasym"),
                      "DG-HRP": hrp_g, **contenders}

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

    for ours in [k for k in ("DG-GMV", "DG-Asym", "DG-HRP") if k in runs]:
        for bench in ("equal_weight",):
            a_l, b_l = runs[ours].fold_losses(), runs[bench].fold_losses()
            n_ = min(len(a_l), len(b_l))
            dm, p = stats.diebold_mariano(a_l[:n_], b_l[:n_])
            dm_rows.append({"N": N, "method": ours, "vs": bench,
                            "dm": round(float(dm), 3), "p": round(float(p), 5)})

pd.DataFrame(rows).to_csv(RESULTS / "sector_sweep_market_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "sector_sweep_market_dm.csv", index=False)
pd.concat(wealth, axis=1).to_csv(RESULTS / "wealth_paths_sweep.csv")
print("\nsector sweep with market benchmark done.")
