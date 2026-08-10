"""Ablation ③: combined significance tests across universes.

Inputs: fold_losses_{tag}.csv from main_experiments runs (top-100 and
top-400). For each learned module vs a baseline:
- per-universe DM (reference)
- pooled DM on the per-date average loss differential across universes
- stationary-bootstrap CI on the mean differential
- sign test across (module x universe) combinations
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from diffgerber.stats import diebold_mariano, stationary_bootstrap

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
TAGS = {"top100": "gs2022", "top400": "gs2022_top400"}
KINDS = ["tglobal", "tasset", "tasym", "tstate"]
BASES = {"fixed": "gerber_gs2022_c0.5", "lw": "ledoit_wolf"}

panels = {}
for uni, tag in TAGS.items():
    df = pd.read_csv(RESULTS / f"fold_losses_{tag}.csv")
    panels[uni] = df.pivot(index="date", columns="method", values="loss")

for base_name, base_col in BASES.items():
    print(f"\n===== vs {base_col} =====")
    sign_neg = 0
    for kind in KINDS:
        diffs = {}
        for uni, P in panels.items():
            pair = P[[f"diffgerber_{kind}", base_col]].dropna()   # paired, aligned
            d = (pair[f"diffgerber_{kind}"] - pair[base_col]).values
            diffs[uni] = d
            dm, p = diebold_mariano(pair[f"diffgerber_{kind}"].values,
                                    pair[base_col].values)
            if d.mean() < 0:
                sign_neg += 1
            print(f"  {kind:8s} {uni:7s}: mean diff={d.mean():+.3e}  DM={dm:+.2f} p={p:.4f}")
        # pooled: average differential across universes per date
        common = panels["top100"].index.intersection(panels["top400"].index)
        pooled = np.mean(
            [(panels[u][f"diffgerber_{kind}"] - panels[u][base_col]).loc[common].values
             for u in panels], axis=0)
        n = len(pooled)
        dbar = pooled.mean()
        se = pooled.std(ddof=1) / np.sqrt(n)
        t = dbar / se
        p = 2 * sps.t.sf(abs(t), df=n - 1)
        lo, med, hi = stationary_bootstrap(pooled, np.mean, n_boot=5000, mean_block=3)
        print(f"  {kind:8s} POOLED : t={t:+.2f} p={p:.4f}  "
              f"boot95%CI=[{lo:+.2e}, {hi:+.2e}] {'<0 OK' if hi < 0 else ''}")
    k = sign_neg
    n_comb = len(KINDS) * len(panels)
    p_sign = sps.binomtest(k, n_comb, 0.5, alternative="greater").pvalue
    print(f"  SIGN TEST: {k}/{n_comb} combos improve; binomial p={p_sign:.4f}")
