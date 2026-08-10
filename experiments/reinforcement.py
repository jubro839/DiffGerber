"""Pre-registered reinforcement experiments (Phase 0 of the writing plan).
Committed before execution; results enter the paper as they come.

R1  Paired stationary-bootstrap tests for the DG-HRP vs plain-HRP headline:
    joint time-resampling of the two net daily series, Delta-Sharpe and
    Delta-Sortino CIs + one-sided p (top-100 and sector N=30). Decides the
    strength of the "improves all five metrics" sentence.
R2  Asymmetric 2-D grid control: hard backtests on (c_up, c_down) in
    {0.5,0.75,1.0,1.25,1.5}^2 (top-100, sector N=100); learned T-asym is
    compared against the ex-post grid best (vol objective + DM).
R3  X3: joint 3-parameter learning (c_up, c_down, delta) — the gradient-vs-
    grid scaling argument (5^3 = 125 grid cells vs one training run).
R4  Estimation-objective contrast: learn c by Frobenius loss to the realized
    out-of-window covariance; compare the learned path and GMV performance
    against the decision-learned c (the DFL existence proof).
R5  Analytic shrinkage baseline: Schafer-Strimmer-style formula intensity
    toward the learned-c Gerber target (delta from a formula, not learned),
    isolating the value of LEARNING delta in X2/DG-Shrink.
R6  Mechanism table: learned c vs universe characteristics (mean |corr|,
    cross-sector pair share, vol dispersion) across all studied universes.

Frozen protocol throughout: window 252/step 21, eval 2017+, 10 bps, annual
expanding refit with horizon-aware boundary, fresh Adam lr 0.02 x 40 epochs,
straight-through, gs2022, scale-relative ridge 1e-3, simple-return metrics.
"""

import json
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
from diffgerber import backtest, baselines, data, stats
from diffgerber.model import tau_schedule
from diffgerber.portfolio_layer import decision_loss, gmv_closed_form
from diffgerber.soft_gerber import SoftGerber, hard_gerber
from diffgerber.threshold_net import AsymThreshold, GlobalThreshold

torch.set_default_dtype(torch.float64)
WINDOW, STEP = 252, 21
EPOCHS = int(os.environ.get("DG_EPOCHS", 40))
LAST_YEAR = int(os.environ.get("DG_LAST_YEAR", 2026))
SECTIONS = set(os.environ.get("DG_SECTIONS", "1,2,3,4,5,6").split(","))
GRID = [0.5, 0.75, 1.0, 1.25, 1.5]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
METRIC_KEYS = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net",
               "max_drawdown", "avg_turnover"]

print("loading data...")
mem = data.sp500_membership()
px = data.prices_total_return("2015-01-01", "2026-07-28",
                              symbols=sorted(mem["fmp_symbol"].unique()))
rets = data.log_returns(px)
dates = rets.index


def softplus(x):
    return math.log1p(math.exp(x))


def load_scalar_thresholds(name):
    raw = json.load(open(RESULTS / name))
    out = {"tglobal": {}, "tasym": {}}
    for k, v in raw.items():
        mod, year = k.rsplit("_", 1)
        if "raw" in v:
            out[mod][int(year)] = (softplus(v["raw"]), None)
        else:
            out[mod][int(year)] = (softplus(v["raw_up"]), softplus(v["raw_down"]))
    return out


TH100 = load_scalar_thresholds("scalar_thresholds_gs2022.json")


def uni_factory(kind):
    _m = {}
    def uni(d):
        if d not in _m:
            if kind == "top100":
                _m[d] = data.pit_universe(d, 100, px, mem)
            else:
                _m[d] = data.pit_universe_sector(d, int(kind[6:]), px, mem)
        return _m[d]
    return uni


def eval_range():
    fold_dates = [dates[i] for i in range(WINDOW, len(dates) - 1, STEP)
                  if dates[i].year <= LAST_YEAR]
    return (min(d for d in fold_dates if d.year >= 2017),
            max(fold_dates) + pd.Timedelta(days=1))


def gerber_G(R, cu, cd=None):
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    return hard_gerber(Rt / s, cu, "gs2022", c_down=cd).numpy(), s.numpy()


def net_simple(run, cost_bps=10.0):
    r_log = run.daily_returns.copy()
    cost = pd.Series(0.0, index=r_log.index)
    for f in run.folds:
        if f.date in cost.index:
            cost.loc[f.date] += f.turnover * cost_bps * 1e-4
    return np.expm1(r_log - cost)


def sr_so(x):
    ann = x.mean() * 252
    sr = ann / (x.std(ddof=1) * np.sqrt(252))
    so = ann / (np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) * np.sqrt(252))
    return sr, so


def paired_boot(a, b, n_boot=2000, block=21, seed=0):
    """Joint stationary bootstrap of two ALIGNED daily series; returns
    observed and bootstrap distribution of (dSR, dSo)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    p = 1.0 / block
    d_sr, d_so = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        sa, so_a = sr_so(a[idx])
        sb, so_b = sr_so(b[idx])
        d_sr[k], d_so[k] = sa - sb, so_a - so_b
    return d_sr, d_so


def fold_builder(uni):
    folds = []
    for i in range(WINDOW, len(dates) - 1, STEP):
        syms = uni(dates[i])
        folds.append({
            "date": dates[i],
            "Rw": torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values),
            "Rf": torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values),
        })
    return folds


def yearly_train(folds, module, fold_loss):
    torch.manual_seed(0)
    per_year = {}
    for year in range(2017, LAST_YEAR + 1):
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
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
        print(f"    {year}: {per_year[year]} ({len(train)} folds, "
              f"{time.time()-t0:.0f}s)", flush=True)
    return per_year


boot_rows, grid_rows, x3_rows, frob_rows, ss_rows, mech_rows = ([] for _ in range(6))
start, end = eval_range()

# ---------------- R1: paired bootstrap for DG-HRP vs plain HRP -------------
if "1" in SECTIONS:
    print("== R1: paired bootstrap A3 vs HRP ==", flush=True)
    for kind, thmap in [("top100", TH100["tglobal"]),
                        ("sector30", None)]:
        uni = uni_factory(kind)
        if thmap is None:
            th = pd.read_csv(RESULTS / "arch_v2b_thresholds_N30.csv")
            thmap = {int(r.year): (float(r.c_up), None)
                     for r in th[th.module == "tglobal"].itertuples()}

        def a3_fn(R, symbols, as_of, thmap=thmap):
            G, _ = gerber_G(R, thmap.get(as_of.year, (0.5, None))[0]
                            if isinstance(thmap.get(as_of.year), tuple)
                            else thmap.get(as_of.year, 0.5))
            return np.asarray(baselines.hrp_weights(R, corr=G))

        ra = backtest.run_walkforward("a3", rets, a3_fn, uni, start, end,
                                      window=WINDOW, step=STEP)
        rh = backtest.run_walkforward("hrp", rets,
                                      lambda R, s, d: baselines.hrp_weights(R),
                                      uni, start, end, window=WINDOW, step=STEP)
        na, nh = net_simple(ra), net_simple(rh)
        obs_sr = sr_so(na.values)[0] - sr_so(nh.values)[0]
        obs_so = sr_so(na.values)[1] - sr_so(nh.values)[1]
        d_sr, d_so = paired_boot(na.values, nh.values)
        for stat, obs, d in [("dSharpe", obs_sr, d_sr), ("dSortino", obs_so, d_so)]:
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_one = float((d <= 0).mean())
            boot_rows.append({"universe": kind, "stat": stat,
                              "observed": round(float(obs), 4),
                              "ci_lo": round(float(lo), 4),
                              "ci_hi": round(float(hi), 4),
                              "p_onesided": round(p_one, 4)})
            print(f"  {kind} {stat}: obs={obs:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                  f"p(<=0)={p_one:.4f}", flush=True)

# ---------------- R2: asymmetric 2-D grid control --------------------------
if "2" in SECTIONS:
    print("== R2: asym 2-D grid ==", flush=True)
    for kind in ["top100", "sector100"]:
        uni = uni_factory(kind)
        runs = {}
        for cu in GRID:
            for cd in GRID:
                def wfn(R, symbols, as_of, cu=cu, cd=cd):
                    G, s = gerber_G(R, cu, cd)
                    return backtest.gmv_from_cov(np.outer(s, s) * G)
                r = backtest.run_walkforward(f"g{cu}_{cd}", rets, wfn, uni,
                                             start, end, window=WINDOW, step=STEP)
                mm = r.metrics(cost_bps=10.0)
                runs[(cu, cd)] = r
                grid_rows.append({"universe": kind, "c_up": cu, "c_down": cd,
                                  "kind": "grid",
                                  **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        # learned T-asym on the same range
        if kind == "top100":
            thmap = TH100["tasym"]
        else:
            th = pd.read_csv(RESULTS / "arch_v2b_thresholds_N100.csv")
            thmap = {int(r.year): (float(r.c_up), float(r.c_down))
                     for r in th[th.module == "tasym"].itertuples()}

        def lfn(R, symbols, as_of, thmap=thmap):
            cu, cd = thmap.get(as_of.year, (0.5, 0.5))
            G, s = gerber_G(R, cu, cd)
            return backtest.gmv_from_cov(np.outer(s, s) * G)
        rl = backtest.run_walkforward("learned", rets, lfn, uni, start, end,
                                      window=WINDOW, step=STEP)
        mm = rl.metrics(cost_bps=10.0)
        grid_rows.append({"universe": kind, "c_up": None, "c_down": None,
                          "kind": "learned",
                          **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
        best = min(((k, r.metrics(cost_bps=10.0)["ann_vol_net"])
                    for k, r in runs.items()), key=lambda t: t[1])
        dm, p = stats.diebold_mariano(rl.fold_losses(),
                                      runs[best[0]].fold_losses())
        grid_rows.append({"universe": kind, "c_up": best[0][0],
                          "c_down": best[0][1], "kind": "expost_best_vs_learned",
                          "ann_vol_net": round(best[1], 5),
                          "sharpe_net": round(float(dm), 3),
                          "sortino_net": round(float(p), 5)})
        print(f"  {kind}: learned vol={mm['ann_vol_net']:.5f}  "
              f"grid-best {best[0]} vol={best[1]:.5f}  DM={dm:+.2f} p={p:.4f}",
              flush=True)

# ---------------- R3: X3 joint (c_up, c_down, delta) -----------------------
class X3(nn.Module):
    def __init__(self):
        super().__init__()
        self.thr = AsymThreshold(0.5)
        self.raw_d = nn.Parameter(torch.tensor(0.0))
        self.layer = SoftGerber("gs2022", straight_through=True)

    def snapshot(self):
        cu, cd = self.thr()
        return (round(float(cu), 4), round(float(cd), 4),
                round(float(torch.sigmoid(self.raw_d)), 4))


def sample_cov_t(Rw):
    Rc = Rw - Rw.mean(0, keepdim=True)
    return Rc.T @ Rc / (Rw.shape[0] - 1)


if "3" in SECTIONS:
    print("== R3: X3 joint 3-param ==", flush=True)
    uni = uni_factory("top100")
    folds = fold_builder(uni)

    def x3_loss(m, f, tau):
        cu, cd = m.thr()
        delta = torch.sigmoid(m.raw_d)
        s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
        G = m.layer(f["Rw"] / s, cu, tau, c_down=cd)
        Sig = delta * (s.unsqueeze(-1) * G * s.unsqueeze(-2)) \
            + (1 - delta) * sample_cov_t(f["Rw"])
        return decision_loss(gmv_closed_form(Sig), f["Rf"])

    x3map = yearly_train(folds, X3(), x3_loss)
    for y, v in x3map.items():
        x3_rows.append({"year": y, "c_up": v[0], "c_down": v[1], "delta": v[2]})

    def x3_fn(R, symbols, as_of):
        cu, cd, d_ = x3map.get(as_of.year, (0.5, 0.5, 0.5))
        G, s = gerber_G(R, cu, cd)
        S = np.cov(R, rowvar=False, ddof=1)
        return backtest.gmv_from_cov(d_ * np.outer(s, s) * G + (1 - d_) * S)

    r = backtest.run_walkforward("X3", rets, x3_fn, uni, start, end,
                                 window=WINDOW, step=STEP)
    mm = r.metrics(cost_bps=10.0)
    x3_rows.append({"year": "METRICS", **{k: round(float(mm[k]), 5)
                                          for k in METRIC_KEYS}})
    print(f"  X3 top100: vol={mm['ann_vol_net']:.5f} SR={mm['sharpe_net']:+.3f} "
          f"mdd={mm['max_drawdown']:.4f}", flush=True)

# ---------------- R4: estimation-objective (Frobenius) contrast ------------
class TGf(nn.Module):
    def __init__(self):
        super().__init__()
        self.thr = GlobalThreshold(0.5)
        self.layer = SoftGerber("gs2022", straight_through=True)

    def snapshot(self):
        return round(float(self.thr()), 4)


if "4" in SECTIONS:
    print("== R4: Frobenius-objective contrast ==", flush=True)
    uni = uni_factory("top100")
    folds = fold_builder(uni)

    def frob_loss(m, f, tau):
        s = f["Rw"].std(0, unbiased=True).clamp_min(1e-8)
        G = m.layer(f["Rw"] / s, m.thr(), tau)
        Sig = s.unsqueeze(-1) * G * s.unsqueeze(-2)
        Sig_real = f["Rf"].T @ f["Rf"] / f["Rf"].shape[0]
        return ((Sig - Sig_real) ** 2).mean()

    fmap = yearly_train(folds, TGf(), frob_loss)
    for y, c in fmap.items():
        frob_rows.append({"year": y, "c_frobenius": c,
                          "c_decision": round(TH100["tglobal"][y][0], 4)})

    def f_fn(R, symbols, as_of):
        G, s = gerber_G(R, fmap.get(as_of.year, 0.5))
        return backtest.gmv_from_cov(np.outer(s, s) * G)

    r = backtest.run_walkforward("frob", rets, f_fn, uni, start, end,
                                 window=WINDOW, step=STEP)
    mm = r.metrics(cost_bps=10.0)
    frob_rows.append({"year": "METRICS", **{k: round(float(mm[k]), 5)
                                            for k in METRIC_KEYS}})
    print(f"  frobenius-c GMV: vol={mm['ann_vol_net']:.5f} "
          f"SR={mm['sharpe_net']:+.3f}", flush=True)

# ---------------- R5: analytic (formula) shrinkage intensity ---------------
if "5" in SECTIONS:
    print("== R5: analytic-delta baseline ==", flush=True)
    uni = uni_factory("top100")
    deltas = []

    def ss_fn(R, symbols, as_of):
        c = TH100["tglobal"].get(as_of.year, (0.5, None))[0]
        G, s = gerber_G(R, c)
        F = np.outer(s, s) * G
        X = R - R.mean(0)
        n = X.shape[0]
        S = X.T @ X / (n - 1)
        W = np.einsum("ti,tj->tij", X, X)
        varS = n / ((n - 1) ** 3) * ((W - W.mean(0)) ** 2).sum(0)
        d_ = float(np.clip(varS.sum() / max(((S - F) ** 2).sum(), 1e-18), 0, 1))
        deltas.append(d_)
        return backtest.gmv_from_cov(d_ * F + (1 - d_) * S)

    r = backtest.run_walkforward("ss_delta", rets, ss_fn, uni, start, end,
                                 window=WINDOW, step=STEP)
    mm = r.metrics(cost_bps=10.0)
    ss_rows.append({"method": "analytic_delta",
                    "delta_mean": round(float(np.mean(deltas)), 4),
                    "delta_min": round(float(np.min(deltas)), 4),
                    "delta_max": round(float(np.max(deltas)), 4),
                    **{k: round(float(mm[k]), 5) for k in METRIC_KEYS}})
    print(f"  analytic delta in [{np.min(deltas):.3f},{np.max(deltas):.3f}] "
          f"mean={np.mean(deltas):.3f}; vol={mm['ann_vol_net']:.5f} "
          f"SR={mm['sharpe_net']:+.3f} mdd={mm['max_drawdown']:.4f}", flush=True)

# ---------------- R6: mechanism table --------------------------------------
if "6" in SECTIONS:
    print("== R6: mechanism (learned c vs universe features) ==", flush=True)
    sec = data.sector_map()

    def features(uni, sample_years=(2018, 2021, 2024)):
        vals = []
        for y in sample_years:
            d = min((x for x in dates if x.year == y), default=None)
            i = dates.get_loc(dates[dates.searchsorted(d)])
            syms = uni(dates[i])
            R = rets[syms].iloc[max(0, i - WINDOW): i].fillna(0.0).values
            if len(R) < WINDOW:
                continue
            C = np.corrcoef(R, rowvar=False)
            off = C[np.triu_indices_from(C, 1)]
            secs = [sec.get(s, "?") for s in syms]
            cross = np.mean([secs[a] != secs[b]
                             for a in range(len(syms))
                             for b in range(a + 1, len(syms))])
            vols = R.std(0)
            vals.append((np.mean(np.abs(off)), np.mean(off), cross,
                         vols.std() / vols.mean()))
        return np.mean(vals, axis=0)

    def mean_c(csv, module, N=None):
        df = pd.read_csv(RESULTS / csv)
        if N is not None:
            df = df[df.N == N]
        df = df[df.module == module]
        late = df[df.year >= 2019]
        return float((late if len(late) else df).c_up.mean())

    _t100_years = [y for y in range(2019, LAST_YEAR + 1)
                   if y in TH100["tglobal"]] or sorted(TH100["tglobal"])
    specs = [
        ("top100", uni_factory("top100"),
         float(np.mean([TH100["tglobal"][y][0] for y in _t100_years]))),
        ("sector30", uni_factory("sector30"),
         mean_c("arch_v2b_thresholds_N30.csv", "tglobal", None)),
        ("sector50", uni_factory("sector50"),
         mean_c("arch_v2b_thresholds_N50.csv", "tglobal", None)),
        ("sector100", uni_factory("sector100"),
         mean_c("arch_v2b_thresholds_N100.csv", "tglobal", None)),
    ]
    try:
        hp = pd.read_csv(RESULTS / "arch_v3_holdout_params.csv")
        hp = hp[(hp.module == "tglobal") & (hp.year >= 2019)]
        full_px = data.prices_total_return("2015-01-01", "2026-07-28")
        full_rets = data.log_returns(full_px)
        for N in (30, 50):
            _m = {}
            def euni(d, N=N, _m=_m):
                if d not in _m:
                    _m[d] = data.pit_universe_sector_exsp(d, N, full_px, mem)
                return _m[d]
            c_h = float(hp[hp.N == N].c_up.mean())
            # features computed on the full panel
            def efeat(uni=euni):
                vals = []
                for y in (2018, 2021, 2024):
                    d = min((x for x in full_rets.index if x.year == y))
                    i = full_rets.index.get_loc(d)
                    syms = uni(full_rets.index[i])
                    R = full_rets[syms].iloc[max(0, i - WINDOW): i].fillna(0.0).values
                    C = np.corrcoef(R, rowvar=False)
                    off = C[np.triu_indices_from(C, 1)]
                    secs = [sec.get(s, "?") for s in syms]
                    cross = np.mean([secs[a] != secs[b]
                                     for a in range(len(syms))
                                     for b in range(a + 1, len(syms))])
                    vols = R.std(0)
                    vals.append((np.mean(np.abs(off)), np.mean(off), cross,
                                 vols.std() / vols.mean()))
                return np.mean(vals, axis=0)
            f = efeat()
            mech_rows.append({"universe": f"exsp{N}", "learned_c": round(c_h, 4),
                              "mean_abs_corr": round(float(f[0]), 4),
                              "mean_corr": round(float(f[1]), 4),
                              "cross_sector_share": round(float(f[2]), 4),
                              "vol_dispersion": round(float(f[3]), 4)})
    except Exception as e:
        print(f"  exsp features skipped: {e}", flush=True)

    for name, uni, c in specs:
        f = features(uni)
        mech_rows.append({"universe": name, "learned_c": round(c, 4),
                          "mean_abs_corr": round(float(f[0]), 4),
                          "mean_corr": round(float(f[1]), 4),
                          "cross_sector_share": round(float(f[2]), 4),
                          "vol_dispersion": round(float(f[3]), 4)})
    mdf = pd.DataFrame(mech_rows)
    print(mdf.to_string(index=False))
    if len(mdf) > 2:
        for col in ["mean_abs_corr", "mean_corr", "cross_sector_share",
                    "vol_dispersion"]:
            r_ = np.corrcoef(mdf.learned_c, mdf[col])[0, 1]
            print(f"  corr(learned_c, {col}) = {r_:+.3f}", flush=True)

for name, rows_ in [("boot", boot_rows), ("asym_grid", grid_rows),
                    ("x3", x3_rows), ("frobenius", frob_rows),
                    ("analytic_delta", ss_rows), ("mechanism", mech_rows)]:
    if rows_:
        pd.DataFrame(rows_).to_csv(RESULTS / f"reinforcement_{name}.csv",
                                   index=False)
print("\nreinforcement experiments done.")
