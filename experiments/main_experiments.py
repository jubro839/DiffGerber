"""Phase 4: threshold-module grid under one walk-forward protocol.

Modules: fixed c=0.5 / T-global / T-asset / T-asym / T-state.
Same protocol as the pilot: point-in-time top-100, window 252, step 21,
annual expanding refit with warm start, eval 2017+, GMV decision loss.

Usage: main_experiments.py [cos|gs2022]
Outputs under experiments/results/: metrics table, DM tests, learned
threshold trajectories (data for the c_t-vs-VIX figure).
"""

import copy
import json
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
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import (
    AsymThreshold, GlobalThreshold, StateThresholdNet, VocabAssetThreshold,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

NORM = sys.argv[1] if len(sys.argv) > 1 else "gs2022"
START_DATA, END_DATA = "2015-01-01", "2026-07-28"
WINDOW, STEP = 252, 21
TOP_N = int(os.environ.get("DG_TOPN", 100))
EVAL_START_YEAR = 2017
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
FEATS = os.environ.get("DG_FEATS", "v1")   # 'v1' (3 features) or 'v2' (7)
HIDDEN = int(os.environ.get("DG_HIDDEN", 32 if FEATS == "v2" else 16))
TAG = NORM + (f"_top{TOP_N}" if TOP_N != 100 else "") + ("_v2" if FEATS == "v2" else "")
print(f"normalization: {NORM}, universe: top-{TOP_N}, feats: {FEATS}, hidden: {HIDDEN}")

# ---------------- data ----------------
print("loading data...")
mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return(START_DATA, END_DATA, symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index

vix = data.index_levels("^VIX", START_DATA, END_DATA)["close_price"]
vix = vix.reindex(dates).ffill()
csd = rets.std(axis=1)  # cross-sectional dispersion, point-in-time
mkt = rets.mean(axis=1)
ewvol = mkt.ewm(halflife=63).std() * np.sqrt(252)
feat_cols = {
    "log_vix": np.log(vix / 20.0),
    "log_csd": np.log(csd.clip(lower=1e-4) / 0.015),
    "log_ewvol": np.log(ewvol.clip(lower=1e-3) / 0.15),
}
if FEATS == "v2":
    tnx = data.index_levels("^TNX", START_DATA, END_DATA)["close_price"]
    irx = data.index_levels("^IRX", START_DATA, END_DATA)["close_price"]
    term = (tnx.reindex(dates).ffill() - irx.reindex(dates).ffill()) / 10.0
    cum = mkt.cumsum()
    drawdown = cum - cum.rolling(252, min_periods=1).max()  # <= 0
    roll_std = rets.rolling(63, min_periods=20).std()
    exc_rate = ((rets / roll_std).abs() > 1.0).mean(axis=1)
    feat_cols.update({
        "term_spread": term,
        "mkt_drawdown": drawdown.clip(lower=-0.6) * 5.0,
        "vix_chg5": np.log(vix / vix.shift(5)).fillna(0.0) * 4.0,
        "exc_rate": (exc_rate - 0.3) * 3.0,
    })
feats_full = pd.DataFrame(feat_cols).fillna(0.0)
N_FEATURES = feats_full.shape[1]

_uni_memo = {}
def universe_fn(as_of):
    if as_of not in _uni_memo:
        _uni_memo[as_of] = data.pit_universe(as_of, TOP_N, px, mem)
    return _uni_memo[as_of]

print("preparing folds...")
reb_idx = list(range(WINDOW, len(dates) - 1, STEP))
folds = []
for i in reb_idx:
    as_of = dates[i]
    syms = universe_fn(as_of)
    folds.append({
        "date": as_of, "syms": syms,
        "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
        "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        "Fw": torch.as_tensor(feats_full.iloc[i - WINDOW : i].values),
    })
vocab = sorted({s for f in folds for s in f["syms"]})
print(f"folds: {len(folds)}, symbol vocab: {len(vocab)}")

# ---------------- threshold modules ----------------
def make_module(kind):
    if kind == "tglobal":
        return GlobalThreshold(0.5)
    if kind == "tasset":
        return VocabAssetThreshold(vocab, 0.5)
    if kind == "tasym":
        return AsymThreshold(0.5)
    if kind == "tstate":
        return StateThresholdNet(n_features=N_FEATURES, n_assets=None, hidden=HIDDEN)
    raise ValueError(kind)


def fold_thresholds(module, kind, fold):
    """Returns (c, c_down_or_None) broadcastable against z (T, N)."""
    if kind == "tglobal":
        return module(), None
    if kind == "tasset":
        return module(fold["syms"]), None
    if kind == "tasym":
        cu, cd = module()
        return cu, cd
    if kind == "tstate":
        return module(fold["Fw"]), None  # (T, 1) broadcasts over assets
    raise ValueError(kind)


LR = {"tglobal": 0.02, "tasset": 0.02, "tasym": 0.02, "tstate": 3e-3}
KINDS = ["tglobal", "tasset", "tasym", "tstate"]

# equal optimization budget for every arm (T-state previously ran at lr=3e-3
# with the same 40 epochs as the lr=0.02 arms, i.e. 6.7x less)
LR = {k: 0.02 for k in LR}
STRAIGHT_THROUGH = os.environ.get("DG_STRAIGHT_THROUGH", "1") == "1"
print(f"straight-through (train==deploy estimator): {STRAIGHT_THROUGH}")

snapshots = {}   # (kind, year) -> state_dict
print("training (annual expanding refit, warm start)...")
for kind in KINDS:
    module = make_module(kind)
    layer = SoftGerber(NORM, straight_through=STRAIGHT_THROUGH)
    for year in range(EVAL_START_YEAR, LAST_YEAR + 1):
        # fresh optimizer each refit: keeping Adam state across years gave the
        # 2026 refit 400 cumulative steps vs 40 for 2017, which confounded the
        # learned-threshold trajectory with step accumulation
        opt = torch.optim.Adam(module.parameters(), lr=LR[kind])
        # exclude folds whose 21-day realization crosses into the eval year
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = [f for f in folds
                 if dates[min(dates.get_loc(f["date"]) + STEP, len(dates) - 1)] < cutoff]
        t0 = time.time()
        for ep in range(EPOCHS):
            tau = tau_schedule(ep, EPOCHS, 0.2, 0.02)
            opt.zero_grad()
            loss = 0.0
            for f in train:
                s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
                c, cd = fold_thresholds(module, kind, f)
                G = layer(f["Rw"] / s, c, tau, c_down=cd)
                Sigma = s.unsqueeze(-1) * G * s.unsqueeze(-2)
                w = gmv_closed_form(Sigma)
                loss = loss + decision_loss(w, f["Rf"])
            (loss / len(train)).backward()
            opt.step()
        snapshots[(kind, year)] = copy.deepcopy(module.state_dict())
        with torch.no_grad():
            c_dbg, cd_dbg = fold_thresholds(module, kind, train[-1])
            c_mean = float(torch.as_tensor(c_dbg).mean())
            extra = f" c_down={float(torch.as_tensor(cd_dbg).mean()):.3f}" if cd_dbg is not None else ""
        print(f"  {kind} {year}: mean c={c_mean:.3f}{extra} "
              f"({len(train)} folds, {time.time() - t0:.0f}s)")

# ---------------- evaluation ----------------
eval_modules = {kind: make_module(kind) for kind in KINDS}

def make_weight_fn(kind):
    def fn(R, symbols, as_of):
        module = eval_modules[kind]
        module.load_state_dict(snapshots[(kind, as_of.year)])
        i = dates.get_loc(as_of)
        fold = {
            "syms": symbols,
            "Fw": torch.as_tensor(feats_full.iloc[i - WINDOW : i].values),
        }
        with torch.no_grad():
            c, cd = fold_thresholds(module, kind, fold)
        Rt = torch.as_tensor(R)
        s = Rt.std(0, unbiased=True).clamp_min(1e-8)
        G = hard_gerber(Rt / s, c, NORM, c_down=cd)
        return backtest.gmv_from_cov((s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy())
    return fn

eval_start = min(f["date"] for f in folds
                 if EVAL_START_YEAR <= f["date"].year <= LAST_YEAR)
eval_end = max(f["date"] for f in folds if f["date"].year <= LAST_YEAR)
contenders = {
    f"gerber_{NORM}_c0.5": backtest.covariance_weight_fn(
        lambda R: baselines.gerber_cov(R, 0.5, NORM)),
    "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
    "hrp": lambda R, s, d: baselines.hrp_weights(R),
}
for kind in KINDS:
    contenders[f"diffgerber_{kind}"] = make_weight_fn(kind)

print("running walk-forward evaluations...")
runs, rows = {}, []
for name, wfn in contenders.items():
    t0 = time.time()
    runs[name] = backtest.run_walkforward(
        name, rets, wfn, universe_fn, eval_start, eval_end + pd.Timedelta(days=1),
        window=WINDOW, step=STEP)
    m = runs[name].metrics(cost_bps=10)
    rows.append({"method": name, **{k: round(float(v), 5) for k, v in m.items()}})
    print(f"  {name:24s} vol_net={m['ann_vol_net']*100:6.2f}%  "
          f"sharpe={m['sharpe_net']:+.2f}  mdd={m['max_drawdown']*100:6.1f}%  "
          f"to={m['avg_turnover']:.2f}  ({time.time() - t0:.0f}s)")

print("\n== Diebold-Mariano vs fixed-c and Ledoit-Wolf ==")
dm_rows = []
for kind in KINDS:
    a = runs[f"diffgerber_{kind}"].fold_losses()
    for base in [f"gerber_{NORM}_c0.5", "ledoit_wolf"]:
        dm, p = stats.diebold_mariano(a, runs[base].fold_losses())
        dm_rows.append({"method": f"diffgerber_{kind}", "vs": base,
                        "dm": round(float(dm), 3), "p": round(float(p), 5)})
        print(f"  {kind:8s} vs {base:22s}: DM={dm:+.2f} p={p:.4f}")

# ---------------- persist results ----------------
pd.DataFrame(rows).to_csv(RESULTS / f"main_metrics_{TAG}.csv", index=False)
pd.DataFrame(dm_rows).to_csv(RESULTS / f"main_dm_{TAG}.csv", index=False)

# persist model snapshots and per-fold losses for downstream analysis
torch.save(snapshots, RESULTS / f"snapshots_{TAG}.pt")
fold_rows = [
    {"method": name, "date": str(f.date.date()), "loss": f.realized_var}
    for name, r in runs.items() for f in r.folds
]
pd.DataFrame(fold_rows).to_csv(RESULTS / f"fold_losses_{TAG}.csv", index=False)

# t-state threshold trajectory over all eval dates (money-figure data)
module = eval_modules["tstate"]
traj = []
for f in folds:
    if not (EVAL_START_YEAR <= f["date"].year <= LAST_YEAR):
        continue
    module.load_state_dict(snapshots[("tstate", f["date"].year)])
    with torch.no_grad():
        c_t = module(f["Fw"])  # (T, 1) over the trailing window
    traj.append({"date": str(f["date"].date()),
                 "c_asof": float(c_t[-1, 0]),
                 "c_window_mean": float(c_t.mean()),
                 "vix": float(vix.loc[f["date"]])})
pd.DataFrame(traj).to_csv(RESULTS / f"tstate_trajectory_{TAG}.csv", index=False)

per_year = {f"{k}_{y}": {kk: vv.tolist() for kk, vv in sd.items()}
            for (k, y), sd in snapshots.items() if k in ("tglobal", "tasym")}
(RESULTS / f"scalar_thresholds_{TAG}.json").write_text(json.dumps(per_year, indent=1))
print(f"\nresults written to {RESULTS}")
