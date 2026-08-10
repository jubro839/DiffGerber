"""Mixed universes, take two: let the threshold differ by asset class.

The first mixed run forced ONE global threshold across sector stocks and a
multi-asset ETF sleeve. Our own findings say that is misspecified: equity
universes want c near 1.0-1.2, ETF/crypto universes want c near 0.1-0.3, and
a single scalar can only compromise (it went to ~0.2 and paid for it on
Sharpe). This experiment removes the misspecification with the paper's own
machinery -- per-asset thresholds broadcast through the same hard-forward /
surrogate-backward construction -- at the coarsest structurally motivated
grouping: one threshold for the equity block, one for the sleeve.

Hypothesis, stated before running: c_eq moves toward the equity region,
c_slv toward the low region, and the split recovers Sharpe the global
compromise gave up while keeping volatility at or below the global learner.

Ceiling first: a 4x4 grid over constant (c_eq, c_slv) measures, ex post, how
much the group structure can add at all. If that surface is flat, no learner
can help and the answer is reported as such.

Arms (frozen protocol throughout -- 252/21, eval 2017+, 10bp, annual warm
refits, Adam 0.02 x 40, tau 0.2 -> 0.02, gs2022, ridge 1e-3, seed 0):

    tglobal          one scalar c            (re-run for aligned fold losses)
    tgroup           c_eq, c_slv             (2 params)
    tglobal_shrink   c + delta               (2 params)
    tgroup_shrink    c_eq, c_slv + delta     (3 params)

Baselines re-run for aligned DM: fixed c=0.5, LW, ANS, HRP, EW.

Outputs: mixed_group_metrics.csv, mixed_group_dm.csv, mixed_group_params.csv
"""

import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber

torch.set_default_dtype(torch.float64)
WINDOW, STEP, EPOCHS = 252, 21, 40
# the paper reports N = 30 and 70; mix50 outputs from the earlier
# sweep are kept in results/ and simply not cited
SIZES = [30, 70]
SLEEVE = ["IWM", "EFA", "EEM", "AGG", "HYG", "TIP", "GLD", "VNQ", "TLT"]
GRID2 = [0.25, 0.5, 1.0, 1.5]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def inv_softplus(y):
    return math.log(math.expm1(y))


class GroupThreshold(nn.Module):
    """One learned threshold per asset group, broadcast to a per-asset vector."""

    def __init__(self, gidx, init=0.5, learn_delta=False):
        super().__init__()
        self.register_buffer("gidx", torch.as_tensor(gidx))
        n_groups = int(self.gidx.max()) + 1
        self.raw = nn.Parameter(torch.full((n_groups,), inv_softplus(init)))
        self.raw_d = nn.Parameter(torch.tensor(0.0)) if learn_delta else None

    def forward(self):
        c = F.softplus(self.raw)[self.gidx]
        d = torch.sigmoid(self.raw_d) if self.raw_d is not None else None
        return c, d

    def group_values(self):
        with torch.no_grad():
            c = F.softplus(self.raw)
            d = float(torch.sigmoid(self.raw_d)) if self.raw_d is not None else None
        return [float(x) for x in c], d


class ScalarThreshold(nn.Module):
    def __init__(self, init=0.5, learn_delta=False):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(inv_softplus(init)))
        self.raw_d = nn.Parameter(torch.tensor(0.0)) if learn_delta else None

    def forward(self):
        d = torch.sigmoid(self.raw_d) if self.raw_d is not None else None
        return F.softplus(self.raw), d

    def group_values(self):
        with torch.no_grad():
            d = float(torch.sigmoid(self.raw_d)) if self.raw_d is not None else None
        return [float(F.softplus(self.raw))], d


def sample_cov_t(Rw):
    Rc = Rw - Rw.mean(0, keepdim=True)
    return Rc.T @ Rc / (Rw.shape[0] - 1)


def train(folds, module, tag):
    torch.manual_seed(0)
    layer = SoftGerber("gs2022", straight_through=True)
    per_year = {}
    for year in range(2017, 2027):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        cutoff = pd.Timestamp(f"{year}-01-01")
        tr = [f for f in folds
              if dates[min(dates.get_loc(f["date"]) + STEP,
                           len(dates) - 1)] < cutoff]
        if not tr:
            per_year[year] = module.group_values()
            continue
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in tr:
                c, delta = module()
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                G = layer(f["Rw"] / s, c, tau)
                Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                if delta is not None:
                    Sig = delta * Sig + (1 - delta) * sample_cov_t(f["Rw"])
                loss = loss + decision_loss(gmv_closed_form(Sig), f["Rf"])
            (loss / len(tr)).backward()
            opt.step()
        per_year[year] = module.group_values()
        cs, d_ = per_year[year]
        d_str = f" delta={d_:.3f}" if d_ is not None else ""
        print(f"    {tag} {year}: c={['%.3f' % x for x in cs]}{d_str} "
              f"({len(tr)} folds, {time.time()-t0:.0f}s)", flush=True)
    return per_year


def make_wfn(per_year, gidx, blend):
    """Deploy hard Gerber at that year's (per-group c, delta)."""
    gidx_t = torch.as_tensor(gidx)

    def fn(R, symbols, as_of):
        cs, d_ = per_year.get(as_of.year, ([0.5], 0.5 if blend else None))
        cvec = torch.as_tensor(cs)[gidx_t] if len(cs) > 1 else float(cs[0])
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, cvec, "gs2022")
        Sig = (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy()
        if blend and d_ is not None:
            Sig = d_ * Sig + (1 - d_) * np.cov(R, rowvar=False, ddof=1)
        return backtest.gmv_from_cov(Sig)
    return fn


def fixed_group_fn(c_eq, c_slv, gidx):
    gidx_t = torch.as_tensor(gidx)

    def fn(R, symbols, as_of):
        cvec = torch.tensor([c_eq, c_slv])[gidx_t]
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, cvec, "gs2022")
        return backtest.gmv_from_cov(
            (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
    return fn


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

rows, dm_rows, par_rows = [], [], []
for N in SIZES:
    uname = f"mix{N}"
    gidx = [0] * N + [1] * len(SLEEVE)
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

    arms = {}
    for tag, module, blend in (
            ("tglobal", ScalarThreshold(), False),
            ("tgroup", GroupThreshold(gidx), False),
            ("tglobal_shrink", ScalarThreshold(learn_delta=True), True),
            ("tgroup_shrink", GroupThreshold(gidx, learn_delta=True), True)):
        print(f"  [{tag}]", flush=True)
        per_year = train(folds, module, tag)
        arms[tag] = make_wfn(per_year, gidx, blend)
        for y, (cs, d_) in sorted(per_year.items()):
            par_rows.append({"universe": uname, "arm": tag, "year": y,
                             "c_eq": round(cs[0], 4),
                             "c_slv": round(cs[1], 4) if len(cs) > 1 else None,
                             "delta": round(d_, 4) if d_ is not None else None})

    contenders = dict(arms)
    contenders["gerber_c0.5"] = lambda R, s_, d_: backtest.covariance_weight_fn(
        lambda R_: baselines.gerber_cov(R_, 0.5, "gs2022"))(R, s_, d_)
    for ce in GRID2:
        for cs_ in GRID2:
            contenders[f"g2_{ce}_{cs_}"] = fixed_group_fn(ce, cs_, gidx)
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
        if not name.startswith("g2_"):
            print(f"  {name:16s} ret={m['ann_ret_net']*100:6.2f}  "
                  f"vol={m['ann_vol_net']*100:6.2f}  SR={m['sharpe_net']:+.3f}  "
                  f"mdd={m['max_drawdown']*100:6.1f}", flush=True)

    PAIRS = [(a, b) for a in arms for b in
             ("gerber_c0.5", "ledoit_wolf", "ans", "equal_weight")]
    PAIRS += [("tgroup", "tglobal"), ("tgroup_shrink", "tglobal_shrink"),
              ("tgroup_shrink", "tgroup")]
    for a, b in PAIRS:
        dm, p = stats.diebold_mariano(runs[a].fold_losses(),
                                      runs[b].fold_losses())
        dm_rows.append({"universe": uname, "method": a, "vs": b,
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "mixed_group_metrics.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "mixed_group_dm.csv", index=False)
pd.DataFrame(par_rows).to_csv(RESULTS / "mixed_group_params.csv", index=False)

print("\n=== ceiling: constant (c_eq, c_slv) surface, ex post ===")
for N in SIZES:
    uname = f"mix{N}"
    s = df[df.universe == uname].set_index("method")
    g2 = s[s.index.str.startswith("g2_")]
    bv, bs = g2.loc[g2.ann_vol_net.idxmin()], g2.loc[g2.sharpe_net.idxmax()]
    diag = g2.loc[[f"g2_{c}_{c}" for c in GRID2]]
    dv = diag.loc[diag.ann_vol_net.idxmin()]
    print(f"{uname}: 2D-best vol {bv.ann_vol_net*100:.2f} ({bv.name})  "
          f"SR {bs.sharpe_net:.3f} ({bs.name})  | diag(best global) vol "
          f"{dv.ann_vol_net*100:.2f} ({dv.name})")
    for k in ("tgroup", "tgroup_shrink", "tglobal", "tglobal_shrink"):
        r = s.loc[k]
        print(f"    {k:16s} vol={r.ann_vol_net*100:.2f} SR={r.sharpe_net:.3f}")
print("\nmixed-group experiment done.")
