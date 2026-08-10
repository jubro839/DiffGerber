"""E3: subperiod decomposition + transaction-cost sensitivity.

Subperiods from stored fold losses; cost table from cheap eval-only reruns
of the scalar-threshold methods (fixed / T-global / T-asym) plus LW.
Note: top-100 tstate column in fold_losses is the overfit v2 variant —
excluded here; T-state subperiod numbers at top-400 use the v2 file too
(slightly worse than v1, i.e. conservative).
"""

import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber import backtest, baselines, data

torch.set_default_dtype(torch.float64)
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

PERIODS = [("2017-19", 2017, 2019), ("2020", 2020, 2020),
           ("2021-22", 2021, 2022), ("2023-26", 2023, 2026)]

def periodize(datestr):
    y = int(datestr[:4])
    for name, lo, hi in PERIODS:
        if lo <= y <= hi:
            return name

# ---- subperiod table from stored fold losses ----
rows = []
for tag, label in [("gs2022", "top100"), ("gs2022_top400", "top400")]:
    P = pd.read_csv(RESULTS / f"fold_losses_{tag}.csv").pivot(
        index="date", columns="method", values="loss")
    P["period"] = [periodize(d) for d in P.index]
    # tstate omitted: only the overfit v2 fold losses were persisted, and the
    # threshold snapshots read below are from the v1 run — mixing them would
    # put two different training runs in one table (see docstring).
    methods = ["gerber_gs2022_c0.5", "ledoit_wolf", "hrp",
               "diffgerber_tglobal", "diffgerber_tasym", "diffgerber_tstate"]
    g = P.groupby("period")[methods].mean().apply(lambda x: np.sqrt(x * 252) * 100)
    g["n_folds"] = P.groupby("period").size()
    g.insert(0, "universe", label)
    rows.append(g.reset_index())
sub = pd.concat(rows)
sub.to_csv(RESULTS / "subperiod_analysis.csv", index=False)
print(sub.round(2).to_string(index=False))

# ---- cost sensitivity (eval-only reruns) ----
mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return("2015-01-01", "2026-07-28", symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index
reb = [dates[i] for i in range(252, len(dates) - 1, 21)]
eval_start = min(d for d in reb if d.year >= 2017)

cost_rows = []
for topn, tag in [(100, "gs2022"), (400, "gs2022_top400")]:
    _m = {}
    def uni(d, topn=topn, _m=_m):
        if d not in _m:
            _m[d] = data.pit_universe(d, topn, px, mem)
        return _m[d]
    snap = json.loads((RESULTS / f"scalar_thresholds_{tag}.json").read_text())
    cg = {int(k.split("_")[1]): math.log(1 + math.exp(snap[k]["raw"]))
          for k in snap if k.startswith("tglobal_")}
    asym = {int(k.split("_")[1]): (math.log(1 + math.exp(snap[k]["raw_up"])),
                                   math.log(1 + math.exp(snap[k]["raw_down"])))
            for k in snap if k.startswith("tasym_")}

    def fn_fixed(R, s_, d_):
        return backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, 0.5, "gs2022"))(R, s_, d_)
    def fn_tglobal(R, s_, d_):
        return backtest.covariance_weight_fn(
            lambda R_: baselines.gerber_cov(R_, cg[d_.year], "gs2022"))(R, s_, d_)
    def fn_tasym(R, s_, d_):
        cu, cd = asym[d_.year]
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        from diffgerber.soft_gerber import hard_gerber
        G = hard_gerber(Rt / s, cu, "gs2022", c_down=cd)
        return backtest.gmv_from_cov((s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
    fn_lw = backtest.covariance_weight_fn(baselines.ledoit_wolf_cov)

    for name, wfn in [("gerber_c0.5", fn_fixed), ("ledoit_wolf", fn_lw),
                      ("tglobal", fn_tglobal), ("tasym", fn_tasym)]:
        r = backtest.run_walkforward(name, rets, wfn, uni, eval_start, dates[-1],
                                     window=252, step=21)
        for bps in [0, 5, 10, 20]:
            m = r.metrics(cost_bps=bps)
            cost_rows.append({"universe": f"top{topn}", "method": name, "cost_bps": bps,
                              "ann_vol_net": round(m["ann_vol_net"], 5),
                              "sharpe_net": round(m["sharpe_net"], 3)})
        print(f"top{topn} {name}: vol@0bps={cost_rows[-4]['ann_vol_net']*100:.2f}% "
              f"sharpe 0/5/10/20bps = "
              f"{[c['sharpe_net'] for c in cost_rows[-4:]]}")

pd.DataFrame(cost_rows).to_csv(RESULTS / "cost_sensitivity.csv", index=False)
print("E3 done.")
