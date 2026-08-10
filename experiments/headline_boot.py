"""Are the headline Sharpe and Sortino margins real, or point estimates?

The Diebold-Mariano tests in the paper are on per-fold realized VARIANCE,
which is the quantity the estimators are trained to minimize. That leaves the
Sharpe and Sortino columns of the headline panel untested: they are reported
as point estimates and nothing says whether the gaps survive sampling noise.

This runs the same paired stationary bootstrap already used on the top-100
universe (reinforcement.py, R1) over the headline universes, resampling the
saved daily net-of-cost paths jointly so the comparison stays paired. No
retraining: the portfolios are fixed, this only asks how much of each gap
would survive a different draw of the same decade.

Protocol is copied from reinforcement.py deliberately -- 2000 replications,
mean block 21 days (one rebalance), seed 0 -- so the numbers are comparable
with the top-100 result already in the paper.

One-sided p is the bootstrap mass at or below zero, i.e. the chance of seeing
no advantage.
"""

import pathlib

import numpy as np
import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SIZES = [30, 50, 70]
N_BOOT, BLOCK, SEED = 2000, 21, 0

# every comparison the manuscript makes on the Sharpe/Sortino columns
PAIRS = [
    ("DG-GMV", "gerber_c0.5"),      # the central claim: learned vs published
    ("DG-GMV", "ledoit_wolf"),
    ("DG-GMV", "ans"),
    ("DG-HRP", "hrp"),              # learned correlation vs sample, same layer
    ("DG-HRP", "equal_weight"),
]


def sr_so(x):
    """Annualized Sharpe and Sortino of a daily simple-return series."""
    m, s = x.mean(), x.std(ddof=1)
    dn = x[x < 0].std(ddof=1) if (x < 0).any() else np.nan
    return (m / s * np.sqrt(252) if s > 0 else np.nan,
            m / dn * np.sqrt(252) if dn and dn > 0 else np.nan)


def paired_boot(a, b, n_boot=N_BOOT, block=BLOCK, seed=SEED):
    """Politis-Romano stationary bootstrap, both series on the same index."""
    rng = np.random.default_rng(seed)
    n, p = len(a), 1.0 / block
    d_sr, d_so = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        sa, soa = sr_so(a[idx])
        sb, sob = sr_so(b[idx])
        d_sr[k], d_so[k] = sa - sb, soa - sob
    return d_sr, d_so


w = pd.read_csv(RESULTS / "headline_wealth.csv", index_col=0)
rets = w.pct_change().dropna()          # wealth path -> daily simple returns
print(f"{len(rets)} daily observations, {len(PAIRS)} pairs x {len(SIZES)} universes")

rows = []
for N in SIZES:
    for a_, b_ in PAIRS:
        a, b = rets[f"N{N}|{a_}"].values, rets[f"N{N}|{b_}"].values
        obs_sr = sr_so(a)[0] - sr_so(b)[0]
        obs_so = sr_so(a)[1] - sr_so(b)[1]
        d_sr, d_so = paired_boot(a, b)
        for stat, obs, d in (("dSharpe", obs_sr, d_sr), ("dSortino", obs_so, d_so)):
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_one = float((d <= 0).mean())
            rows.append({"N": N, "method": a_, "vs": b_, "stat": stat,
                         "observed": round(float(obs), 4),
                         "ci_lo": round(float(lo), 4),
                         "ci_hi": round(float(hi), 4),
                         "p_onesided": round(p_one, 4)})
        print(f"  N={N} {a_} vs {b_:12s} dSR={obs_sr:+.3f} p={rows[-2]['p_onesided']:.3f}"
              f"   dSo={obs_so:+.3f} p={rows[-1]['p_onesided']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "headline_boot.csv", index=False)

print("\n=== how many Sharpe margins survive? ===")
s = df[df.stat == "dSharpe"]
for lvl in (0.05, 0.10):
    k = int(((s.p_onesided < lvl) & (s.observed > 0)).sum())
    print(f"  p < {lvl:.2f}: {k}/{len(s)} comparisons")
print("\nheadline bootstrap done.")
