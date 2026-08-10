"""Architecture v3 — HOLDOUT confirmation (ex-S&P sector universes), run ONCE.

Committed BEFORE arch_v3_dev.py executes. Reads the mechanically selected
winner from arch_v3_winner.csv (zero human discretion between dev results and
this test), re-runs the LEARNING PROCEDURE (not the parameters) on universes
never touched by any prior experiment:

  - Universe: data.pit_universe_sector_exsp — sector-balanced N in {30, 50}
    from names that were NEVER S&P members; same eligibility rules, market
    caps strictly before as_of. Survivorship caveat (disclosed): warehouse
    coverage of ex-S&P delistings is thin before 2020, shared by all methods;
    a from-2020 robustness slice is reported alongside the full window.
  - Training on holdout: T-global c (variance DFL) and X2 (c, delta) under
    the frozen protocol (annual expanding refit, horizon-aware boundary,
    fresh Adam lr 0.02 x 40 epochs, straight-through, gs2022, ridge 1e-3);
    BL_LEARN's alpha (downside DFL) likewise if selected.
  - Evaluated: winner per N + its component singles + the 4 baselines,
    5-metric panel at {0, 10, 25, 50} bps + DM tests + from-2020 slice.

The "uniform superiority" claim is made ONLY if the winner takes 20/20 cells
here; otherwise the achieved cells are reported as-is.
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
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50").split(",")]
COSTS = [0.0, 10.0, 25.0, 50.0]
LINKMAP = {"single": "single", "avg": "average", "ward": "ward"}
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

METRIC_DIR = {"ann_ret_net": 1, "ann_vol_net": -1, "sharpe_net": 1,
              "sortino_net": 1, "max_drawdown": 1}
METRIC_KEYS = list(METRIC_DIR) + ["avg_turnover"]
BASE = ["gerber_c0.5", "ledoit_wolf", "ans", "hrp"]

print("loading FULL price panel (ex-S&P pool included)...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28")   # all symbols
rets = data.log_returns(px)
dates = rets.index

winners = {int(r.N): str(r.winner)
           for r in pd.read_csv(RESULTS / "arch_v3_winner.csv").itertuples()}
print("winners from dev:", winners)


class ShrinkGerber(nn.Module):
    def __init__(self):
        super().__init__()
        self.thr = GlobalThreshold(0.5)
        self.raw_d = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        return self.thr(), torch.sigmoid(self.raw_d)


def sample_cov_t(Rw):
    Rc = Rw - Rw.mean(0, keepdim=True)
    return Rc.T @ Rc / (Rw.shape[0] - 1)


def yearly_train(folds, make_module, fold_loss):
    """Frozen annual-expanding-refit loop; returns {year: snapshot(module)}."""
    torch.manual_seed(0)
    module = make_module()
    per_year = {}
    for year in range(2017, LAST_YEAR + 1):
        opt = torch.optim.Adam([p for p in module.parameters()], lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = [f for f in folds
                 if dates[min(dates.get_loc(f["date"]) + STEP,
                              len(dates) - 1)] < cutoff]
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in train:
                loss = loss + fold_loss(module, f, tau)
            (loss / len(train)).backward()
            opt.step()
        with torch.no_grad():
            per_year[year] = module.snapshot()
        print(f"  {module.__class__.__name__} {year}: {per_year[year]} "
              f"({len(train)} folds, {time.time()-t0:.0f}s)", flush=True)
    return per_year


class TG(nn.Module):
    def __init__(self):
        super().__init__()
        self.thr = GlobalThreshold(0.5)
        self.layer = SoftGerber("gs2022", straight_through=True)

    def snapshot(self):
        return round(float(self.thr()), 4)


class XS(ShrinkGerber):
    def __init__(self):
        super().__init__()
        self.layer = SoftGerber("gs2022", straight_through=True)

    def snapshot(self):
        c, d = self()
        return (round(float(c), 4), round(float(d), 4))


def tg_loss(module, f, tau):
    s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
    G = module.layer(f["Rw"] / s, module.thr(), tau)
    Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
    return decision_loss(gmv_closed_form(Sig), f["Rf"])


def xs_loss(module, f, tau):
    c, delta = module()
    s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
    G = module.layer(f["Rw"] / s, c, tau)
    Sig = delta * (s.unsqueeze(-1) * G * s.unsqueeze(-2)) \
        + (1.0 - delta) * sample_cov_t(f["Rw"])
    return decision_loss(gmv_closed_form(Sig), f["Rf"])


rows, dm_rows, p_rows = [], [], []

for N in SIZES:
    print(f"===== holdout N={N} =====", flush=True)
    _m = {}
    def uni(d, N=N, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe_sector_exsp(d, N, px, mem)
        return _m[d]

    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        folds.append({
            "date": dates[i], "syms": syms,
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        })
    last = folds[-1]
    sec = data.sector_map()
    from collections import Counter
    print(f"  sample universe {last['date'].date()}: "
          f"{dict(Counter(sec[s] for s in last['syms']))}", flush=True)
    print(f"  members: {', '.join(last['syms'][:15])} ...", flush=True)

    g_map = yearly_train(folds, TG, tg_loss)          # {year: c}
    x2_map = yearly_train(folds, XS, xs_loss)         # {year: (c, delta)}
    for y in g_map:
        p_rows.append({"N": N, "module": "tglobal", "year": y,
                       "c_up": g_map[y], "delta": None, "alpha": None})
        p_rows.append({"N": N, "module": "X2_shrink", "year": y,
                       "c_up": x2_map[y][0], "delta": x2_map[y][1],
                       "alpha": None})

    def g_c(year):
        return g_map.get(year, 0.5)

    def x2_cd(year):
        return x2_map.get(year, (0.5, 0.5))

    def gerber_corr(R, c):
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        return hard_gerber(Rt / s, c, "gs2022").numpy(), s.numpy()

    def x2_sigma(R, year):
        c, delta = x2_cd(year)
        G, s = gerber_corr(R, c)
        S = np.cov(R, rowvar=False, ddof=1)
        return delta * (np.outer(s, s) * G) + (1.0 - delta) * S

    def gmv_G(R, symbols, as_of):
        G, s = gerber_corr(R, g_c(as_of.year))
        return backtest.gmv_from_cov(np.outer(s, s) * G)

    def gmv_X2(R, symbols, as_of):
        return backtest.gmv_from_cov(x2_sigma(R, as_of.year))

    def hrp_G(link):
        def fn(R, symbols, as_of):
            G, _ = gerber_corr(R, g_c(as_of.year))
            return np.asarray(baselines.hrp_weights(
                R, corr=G, linkage_method=LINKMAP[link]))
        return fn

    def hrp_X2(link):
        def fn(R, symbols, as_of):
            Sig = x2_sigma(R, as_of.year)
            d = np.sqrt(np.diag(Sig))
            C = Sig / np.outer(d, d)
            np.fill_diagonal(C, 1.0)
            return np.asarray(baselines.hrp_weights(
                R, corr=C, linkage_method=LINKMAP[link]))
        return fn

    comp = {"GMV_X2": gmv_X2, "GMV_G": gmv_G,
            **{f"HRP_{l}_G": hrp_G(l) for l in LINKMAP},
            **{f"HRP_{l}_X2": hrp_X2(l) for l in ["avg", "ward"]}}

    def train_alpha(link):
        pre = []
        for f in folds:
            R = f["Rw"].numpy()
            pre.append({"date": f["date"],
                        "wg": torch.as_tensor(gmv_X2(R, f["syms"], f["date"])),
                        "wh": torch.as_tensor(
                            comp[f"HRP_{link}_G"](R, f["syms"], f["date"])),
                        "Rf": f["Rf"]})
        per_year = {}
        for year in range(2017, LAST_YEAR + 1):
            cutoff = pd.Timestamp(f"{year}-01-01")
            train = [p for p in pre
                     if dates[min(dates.get_loc(p["date"]) + STEP,
                                  len(dates) - 1)] < cutoff]
            theta = torch.zeros((), requires_grad=True)
            opt = torch.optim.Adam([theta], lr=0.02)
            for _ in range(EPOCHS):
                opt.zero_grad()
                loss = 0.0
                for p in train:
                    a = torch.sigmoid(theta)
                    w = a * p["wg"] + (1 - a) * p["wh"]
                    port = p["Rf"] @ w
                    loss = loss + (torch.clamp(port, max=0.0) ** 2).mean()
                (loss / len(train)).backward()
                opt.step()
            with torch.no_grad():
                per_year[year] = float(torch.sigmoid(theta))
        return per_year

    def build(name):
        """winner-name -> weight fn (holdout-trained parameters only)."""
        if name in comp:
            return comp[name], [name]
        if name == "BL_LEARN":
            alpha = train_alpha("single")
            for y, a in alpha.items():
                p_rows.append({"N": N, "module": "BL_LEARN", "year": y,
                               "c_up": None, "delta": None,
                               "alpha": round(a, 4)})

            def fn(R, symbols, as_of):
                a = alpha[as_of.year]
                return a * np.asarray(gmv_X2(R, symbols, as_of)) + \
                    (1 - a) * np.asarray(comp["HRP_single_G"](R, symbols, as_of))
            return fn, ["GMV_X2", "HRP_single_G"]
        if name.startswith("BL"):                     # BL{a}_{link}
            a_str, link = name[2:].split("_")
            a = float(a_str)

            def fn(R, symbols, as_of, a=a, link=link):
                return a * np.asarray(gmv_X2(R, symbols, as_of)) + \
                    (1 - a) * np.asarray(comp[f"HRP_{link}_G"](R, symbols, as_of))
            return fn, ["GMV_X2", f"HRP_{link}_G"]
        raise ValueError(f"unknown winner name: {name}")

    wname = winners[N]
    wfn, comps = build(wname)
    todo = {f"WINNER_{wname}": wfn}
    todo.update({c: comp[c] for c in comps if c in comp})
    todo.update({
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
        "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
        "ans": backtest.covariance_weight_fn(
            baselines.analytical_nonlinear_shrinkage),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
    })

    fold_dates = [f["date"] for f in folds if f["date"].year <= LAST_YEAR]
    report_start = min(d for d in fold_dates if d.year >= 2017)
    run_end = max(fold_dates) + pd.Timedelta(days=1)

    runs = {}
    for name, fn in todo.items():
        r = backtest.run_walkforward(name, rets, fn, uni, report_start,
                                     run_end, window=WINDOW, step=STEP)
        runs[name] = r
        for cb in COSTS:
            mm = r.metrics(cost_bps=cb)
            rows.append({"N": N, "method": name, "window": "full_2017_26",
                         "cost_bps": cb,
                         **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        # from-2020 robustness slice (survivorship-safe zone), 10 bps
        r_log = r.daily_returns.copy()
        cost = pd.Series(0.0, index=r_log.index)
        for f in r.folds:
            if f.date in cost.index:
                cost.loc[f.date] += f.turnover * 10.0 * 1e-4
        seg = (r_log - cost).loc["2020-01-01":]
        simple = np.expm1(seg)
        ann_ret = simple.mean() * 252
        ann_vol = simple.std(ddof=1) * np.sqrt(252)
        dn = np.sqrt(np.mean(np.minimum(simple, 0.0) ** 2)) * np.sqrt(252)
        curve = np.exp(seg.cumsum())
        rows.append({"N": N, "method": name, "window": "from_2020",
                     "cost_bps": 10.0,
                     "ann_ret_net": round(float(ann_ret), 5),
                     "ann_vol_net": round(float(ann_vol), 5),
                     "sharpe_net": round(float(ann_ret / ann_vol), 5),
                     "sortino_net": round(float(ann_ret / dn), 5),
                     "max_drawdown": round(float((curve / curve.cummax() - 1).min()), 5),
                     "avg_turnover": None})
        mm = r.metrics(cost_bps=10.0)
        print(f"  N={N} {name:22s} ret={mm['ann_ret_net']*100:6.2f}%  "
              f"vol={mm['ann_vol_net']*100:6.2f}%  SR={mm['sharpe_net']:+.2f}  "
              f"So={mm['sortino_net']:+.2f}  mdd={mm['max_drawdown']*100:6.1f}%",
              flush=True)

    wkey = f"WINNER_{wname}"
    cells, lost = 0, []
    pw = runs[wkey].metrics(cost_bps=10.0)
    for b in BASE:
        pb = runs[b].metrics(cost_bps=10.0)
        for m, d_ in METRIC_DIR.items():
            if d_ * (pw[m] - pb[m]) > 0:
                cells += 1
            else:
                lost.append(f"{b}:{m}")
        a_l, b_l = runs[wkey].fold_losses(), runs[b].fold_losses()
        n_ = min(len(a_l), len(b_l))
        dm, p = stats.diebold_mariano(a_l[:n_], b_l[:n_])
        dm_rows.append({"N": N, "method": wkey, "vs": b,
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})
    print(f"  >>> HOLDOUT N={N}: {wname} wins {cells}/20 cells"
          + (f"; lost: {';'.join(lost)}" if lost else " — UNIFORM"), flush=True)
    dm_rows.append({"N": N, "method": wkey, "vs": "CELLS", "dm": cells,
                    "p": None})

pd.DataFrame(rows).to_csv(RESULTS / "arch_v3_holdout_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "arch_v3_holdout_dm.csv", index=False)
pd.DataFrame(p_rows).to_csv(RESULTS / "arch_v3_holdout_params.csv", index=False)
print("\narch v3 holdout done.")
