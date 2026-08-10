"""Which component of DG-Shrink earns its keep: the threshold c, or delta?

DG-Shrink learns two scalars jointly -- the Gerber threshold c and the
shrinkage intensity delta blending the Gerber covariance with the sample
covariance. The headline panel shows the pair working together; it cannot say
which one carries the risk reduction. This factorial fills the grid on the
three headline universes under the frozen protocol:

    c = 0.5 fixed,  delta = 1 fixed      published Gerber   (Table 1 row)
    c learned,      delta = 1 fixed      DG-GMV             (Table 1 row)
    c = 0.5 fixed,  delta learned        NEW: delta alone
    c learned,      delta fixed 0.5      NEW: c alone, mid blend
    c learned,      delta fixed 0.25     NEW: grid point
    c learned,      delta fixed 0.75     NEW: grid point
    c = 0.5 fixed,  delta = 0.5 fixed    NEW: static blend, nothing learned
    c learned,      delta learned        DG-Shrink          (Table 1 row)

Training is bit-for-bit the DG-Shrink loop of headline_pack.py (seed 0,
warm-started module across years, fresh Adam lr 0.02 x 40 epochs, tau
0.2 -> 0.02, straight-through estimator) with the frozen component held as a
constant rather than a parameter. As a wiring check, DG-Shrink itself is
re-evaluated here from its saved (c, delta) path and must reproduce the
headline metrics exactly.

Outputs:
    ablation_c_delta.csv         five-metric panel, all configs x N, 10bp
    ablation_c_delta_dm.csv      DM on per-fold realized variance, key pairs
    ablation_c_delta_params.csv  learned parameter paths per config
"""

import math
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SIZES = [int(x) for x in os.environ.get("DG_SIZES", "30,50,70").split(",")]
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


class AblShrink(nn.Module):
    """DG-Shrink with either component optionally frozen at a constant."""

    def __init__(self, learn_c=True, learn_delta=True, c0=0.5, d0=0.5):
        super().__init__()
        if learn_c:
            self.thr = GlobalThreshold(c0)
        else:
            self.register_buffer("c_const", torch.tensor(float(c0)))
        if learn_delta:
            # sigmoid(logit(d0)) == d0, so training starts exactly at d0
            self.raw_d = nn.Parameter(torch.tensor(math.log(d0 / (1 - d0))))
        else:
            self.register_buffer("d_const", torch.tensor(float(d0)))

    def forward(self):
        c = self.thr() if hasattr(self, "thr") else self.c_const
        d = (torch.sigmoid(self.raw_d) if hasattr(self, "raw_d")
             else self.d_const)
        return c, d


def sample_cov_t(Rw):
    Rc = Rw - Rw.mean(0, keepdim=True)
    return Rc.T @ Rc / (Rw.shape[0] - 1)


def train(folds, learn_c, learn_delta, d0=0.5, tag=""):
    """The DG-Shrink loop of headline_pack.py, component freeze aside."""
    torch.manual_seed(0)
    module = AblShrink(learn_c, learn_delta, c0=0.5, d0=d0)
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
        print(f"    {tag} {year}: c={per_year[year][0]:.4f} "
              f"delta={per_year[year][1]:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    return per_year


def blend_fn(cmap):
    """Deployed portfolio: hard Gerber at (c_y, delta_y), blended, GMV."""
    def fn(R, symbols, as_of):
        c, delta = cmap.get(as_of.year, (0.5, 0.5))
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, "gs2022").numpy()
        s = s.numpy()
        S = np.cov(R, rowvar=False, ddof=1)
        return backtest.gmv_from_cov(delta * np.outer(s, s) * G + (1 - delta) * S)
    return fn


hm = pd.read_csv(RESULTS / "headline_metrics.csv")
ht = pd.read_csv(RESULTS / "headline_thresholds.csv")

rows, dm_rows, param_rows = [], [], []
for N in SIZES:
    print(f"===== sector N={N} =====", flush=True)
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
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW:i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i:i + STEP].fillna(0.0).values),
        })

    reuse = os.environ.get("DG_REUSE") == "1"
    if reuse:
        # deterministic training already ran; reload its saved paths so a DM
        # extension does not pay for retraining
        pp = pd.read_csv(RESULTS / "ablation_c_delta_params.csv")
        def saved(cfg, N=N, pp=pp):
            q = pp[(pp.N == N) & (pp.config == cfg)]
            return {int(r.year): (float(r.c), float(r.delta))
                    for r in q.itertuples()}
        p_delta, p_c05 = saved("delta_only"), saved("c_only_d0.50")
        p_c025, p_c075 = saved("c_only_d0.25"), saved("c_only_d0.75")
        print("  [reusing saved parameter paths]", flush=True)
    else:
        print("  [delta_only]", flush=True)
        p_delta = train(folds, learn_c=False, learn_delta=True, tag="d-only")
        print("  [c_only, delta=0.5]", flush=True)
        p_c05 = train(folds, learn_c=True, learn_delta=False, d0=0.5, tag="c-only")
        print("  [c_only, delta=0.25]", flush=True)
        p_c025 = train(folds, learn_c=True, learn_delta=False, d0=0.25, tag="c-.25")
        print("  [c_only, delta=0.75]", flush=True)
        p_c075 = train(folds, learn_c=True, learn_delta=False, d0=0.75, tag="c-.75")

    # DG-Shrink from its saved headline path: the wiring check
    xp = ht[(ht.N == N) & (ht.arm == "DG-Shrink") & ht.delta.notna()]
    p_full = {int(r.year): (float(r.c_up), float(r.delta))
              for r in xp.itertuples()}
    p_static = {y: (0.5, 0.5) for y in range(2017, LAST_YEAR + 1)}

    configs = {
        "delta_only": p_delta,
        "c_only_d0.50": p_c05,
        "c_only_d0.25": p_c025,
        "c_only_d0.75": p_c075,
        "static_blend": p_static,
        "DG-Shrink": p_full,
    }
    runs = {}
    for name, cmap in configs.items():
        r = backtest.run_walkforward(name, rets, blend_fn(cmap), uni,
                                     report_start, run_end,
                                     window=WINDOW, step=STEP)
        runs[name] = r
        m = r.metrics(cost_bps=10.0)
        rows.append({"N": N, "config": name,
                     **{k: round(float(m[k]), 5) for k in METRIC_KEYS}})
        for y, (c, d_) in sorted(cmap.items()):
            param_rows.append({"N": N, "config": name, "year": y,
                               "c": round(c, 4), "delta": round(d_, 4)})
        print(f"  {name:14s} vol={m['ann_vol_net']*100:6.3f}%  "
              f"SR={m['sharpe_net']:+.3f}", flush=True)

    # wiring check: the re-evaluated DG-Shrink must equal the headline row
    ours = [r_ for r_ in rows if r_["N"] == N and r_["config"] == "DG-Shrink"][0]
    ref = hm[(hm.N == N) & (hm.method == "DG-Shrink")
             & (hm.cost_bps == 10)].iloc[0]
    diff = max(abs(ours[k] - round(float(ref[k]), 5)) for k in METRIC_KEYS)
    print(f"  wiring check vs headline DG-Shrink: max metric diff {diff:.2e}",
          flush=True)
    # headline_thresholds stores the path rounded to 4 decimals; at N=70 the
    # headline metrics came from the unrounded in-memory parameters, so the
    # re-evaluation can differ by the rounding, never by more
    assert diff < 2e-3, "re-evaluated DG-Shrink diverges beyond param rounding"

    # attribution tests on the loss the components are trained to minimize
    import itertools
    PAIRS = list(itertools.combinations(configs, 2))
    for a, b in PAIRS:
        dm_, p_ = stats.diebold_mariano(runs[a].fold_losses(),
                                        runs[b].fold_losses())
        dm_rows.append({"N": N, "method": a, "vs": b,
                        "dm": round(float(dm_), 3), "p": round(float(p_), 5)})
        print(f"  DM {a} vs {b}: {dm_:+.2f} (p={p_:.4f})", flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "ablation_c_delta.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / "ablation_c_delta_dm.csv", index=False)
pd.DataFrame(param_rows).to_csv(RESULTS / "ablation_c_delta_params.csv",
                                index=False)
print("\nc/delta ablation done.")
