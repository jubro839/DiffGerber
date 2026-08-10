"""The two family arms the mixed run was missing: DG-Asym and DG-HRP.

mixed_group_experiment.py covers the threshold-vector factorial (scalar or
grouped c, with or without the shrinkage blend). Table 1 Panel A also
reports DG-Asym, which learns separate up and down thresholds, and DG-HRP,
which feeds the learned hard correlation to the hierarchical allocation
rule instead of the minimum-variance layer. Neither had been run on the
mixed universes, so the two panels carried different arms for no reason
other than what had been executed. This closes that gap.

Definitions follow Panel A exactly:

    DG-Asym   learn (c_up, c_down) -> hard Gerber -> GMV
    DG-HRP    reuse DG-GMV's learned threshold, hand the hard correlation
              to baselines.hrp_weights (the same rule the HRP comparator
              uses, so the only change is the correlation matrix)

DG-HRP therefore needs no training of its own: it reads the per-year
thresholds the tglobal arm already learned, from mixed_group_params.csv.

Protocol is frozen as everywhere else: window 252 / step 21, evaluation
2017+, 10bp costs, annual expanding refit with warm start, Adam 0.02 x 40
epochs, tau 0.2 -> 0.02, gs2022, scale-relative ridge 1e-3, seed 0.

Outputs: mixed_arms_metrics.csv, mixed_arms_dm.csv, mixed_arms_params.csv
"""

import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber

torch.set_default_dtype(torch.float64)
WINDOW, STEP, EPOCHS = 252, 21, 40
SIZES = [30, 70]
SLEEVE = ["IWM", "EFA", "EEM", "AGG", "HYG", "TIP", "GLD", "VNQ", "TLT"]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def inv_softplus(y):
    return math.log(math.expm1(y))


class AsymThreshold(nn.Module):
    """Separate up/down thresholds, initialised at the published value."""

    def __init__(self, init=0.5):
        super().__init__()
        self.raw_up = nn.Parameter(torch.tensor(inv_softplus(init)))
        self.raw_down = nn.Parameter(torch.tensor(inv_softplus(init)))

    def forward(self):
        return F.softplus(self.raw_up), F.softplus(self.raw_down)


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

# DG-HRP reuses the threshold DG-GMV learned, exactly as in Panel A
_par = pd.read_csv(RESULTS / "mixed_group_params.csv")
TGLOBAL = {(r.universe, int(r.year)): float(r.c_eq)
           for r in _par[_par.arm == "tglobal"].itertuples()}


def train_asym(folds, tag):
    torch.manual_seed(0)
    module = AsymThreshold()
    layer = SoftGerber("gs2022", straight_through=True)
    per_year = {}
    for year in range(2017, 2027):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        tr = [f for f in folds
              if dates[min(dates.get_loc(f["date"]) + STEP,
                           len(dates) - 1)] < cutoff]
        if not tr:
            cu, cd = module()
            per_year[year] = (float(cu), float(cd))
            continue
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in tr:
                cu, cd = module()
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, cu, tau, c_down=cd)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(tr)).backward()
            opt.step()
        with torch.no_grad():
            cu, cd = module()
        per_year[year] = (float(cu), float(cd))
        print(f"    {tag} {year}: c_up={cu:.3f} c_down={cd:.3f} "
              f"({len(tr)} folds, {time.time()-t0:.0f}s)", flush=True)
    return per_year


def asym_wfn(per_year):
    def fn(R, symbols, as_of):
        cu, cd = per_year.get(as_of.year, (0.5, 0.5))
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, cu, "gs2022", c_down=cd)
        return backtest.gmv_from_cov(
            (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
    return fn


def hrp_wfn(uname):
    """Panel A's DG-HRP: learned correlation into the same hierarchy."""
    def fn(R, symbols, as_of):
        c = TGLOBAL.get((uname, as_of.year), 0.5)
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022").numpy()
        return np.asarray(baselines.hrp_weights(R, corr=G))
    return fn


rows, dm_rows, par_rows = [], [], []
for N in SIZES:
    uname = f"mix{N}"
    print(f"===== {uname} ({N}+{len(SLEEVE)} assets) =====", flush=True)
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

    print("  [tasym]", flush=True)
    asym_years = train_asym(folds, uname)
    for y, (cu, cd) in sorted(asym_years.items()):
        par_rows.append({"universe": uname, "arm": "tasym", "year": y,
                         "c_up": round(cu, 4), "c_down": round(cd, 4)})

    contenders = {
        "tasym": asym_wfn(asym_years),
        "hrp_gerber": hrp_wfn(uname),
        "gerber_c0.5": backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, 0.5, "gs2022")),
        "hrp": lambda R, s, d: baselines.hrp_weights(R),
    }

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
        print(f"  {name:12s} ret={m['ann_ret_net']*100:6.2f}%  "
              f"vol={m['ann_vol_net']*100:6.2f}%  SR={m['sharpe_net']:+.3f}  "
              f"mdd={m['max_drawdown']*100:6.1f}%", flush=True)

    for a, b in (("tasym", "gerber_c0.5"), ("hrp_gerber", "gerber_c0.5"),
                 ("hrp_gerber", "hrp")):
        dm, p = stats.diebold_mariano(runs[a].fold_losses(),
                                      runs[b].fold_losses())
        dm_rows.append({"universe": uname, "method": a, "vs": b,
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})
        print(f"  DM {a} vs {b}: {dm:+.2f} (p={p:.4f})", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "mixed_arms_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "mixed_arms_dm.csv", index=False)
pd.DataFrame(par_rows).to_csv(RESULTS / "mixed_arms_params.csv", index=False)
print("\nmixed-universe DG-Asym / DG-HRP done.")
