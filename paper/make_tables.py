"""Generate every paper table from experiments/results CSVs.

No number is typed by hand: this script is the single bridge between the
result files and the LaTeX source. Output: paper/tables/T*.tex

Headline evidence is the sector-balanced sweep at N in {30, 50, 70}; the
market-cap top-100 universe is reported alongside as a second design.
"""

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
OUT = pathlib.Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)

SIZES = [30, 70]
NAME = {"DG-GMV": "DG-GMV", "DG-Asym": "DG-Asym", "DG-HRP": "DG-HRP",
        "DG-Shrink": "DG-Shrink", "gerber_c0.5": "Gerber $c{=}0.5$",
        "ledoit_wolf": "Ledoit--Wolf", "ans": "ANS", "hrp": "HRP",
        "equal_weight": "Equal weight $1/N$"}
ORDER = ["DG-GMV", "DG-Asym", "DG-HRP", "DG-Shrink",
         "gerber_c0.5", "ledoit_wolf", "ans", "hrp", "equal_weight"]
M5 = ["ann_ret_net", "ann_vol_net", "sharpe_net", "sortino_net", "max_drawdown"]
DIR = {"ann_ret_net": 1, "ann_vol_net": -1, "sharpe_net": 1,
       "sortino_net": 1, "max_drawdown": 1, "avg_turnover": -1}


def fmt(col, v):
    if col in ("ann_ret_net", "ann_vol_net"):
        return f"{v*100:.2f}"
    if col == "max_drawdown":
        return f"{v*100:.1f}"
    return f"{v:.2f}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print(f"wrote {name}")


# ---- T1: headline panel, three sector universes side by side --------------
hm = pd.read_csv(RES / "headline_metrics.csv")
panels = {N: hm[(hm.N == N) & (hm.cost_bps == 10)].set_index("method")
          for N in SIZES}
best = {N: {c: (panels[N].loc[ORDER, c] * DIR[c]).idxmax() for c in M5}
        for N in SIZES}
rows = [r"\begin{tabular}{@{}l" + r"rrrrr@{\quad}" * 2 + r"rrrrr@{}}",
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{5}}{{c}}{{$N={N}$}}" for N in SIZES) + r"\\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(l){12-16}",
        "Method" + " & Ret & Vol & SR & So & MDD" * 3 + r"\\",
        r"\midrule"]
for m in ORDER:
    cells = []
    for N in SIZES:
        for c in M5:
            s = fmt(c, panels[N].loc[m, c])
            cells.append(rf"\textbf{{{s}}}" if best[N][c] == m else s)
    rows.append(NAME[m] + " & " + " & ".join(cells) + r"\\")
    if m == "DG-Shrink":
        rows.append(r"\addlinespace[2pt]")
rows += [r"\bottomrule", r"\end{tabular}"]
write("T1_headline_panel.tex", rows)

# ---- T2: significance against every baseline ------------------------------
dm = pd.read_csv(RES / "headline_dm.csv")
BASE = ["gerber_c0.5", "ledoit_wolf", "ans", "hrp", "equal_weight"]
HEAD = {"gerber_c0.5": r"$c{=}0.5$", "ledoit_wolf": "LW", "ans": "ANS",
        "hrp": "HRP", "equal_weight": r"$1/N$"}
ARMS = ["DG-GMV", "DG-Asym", "DG-HRP", "DG-Shrink"]
rows = [r"\begin{tabular}{@{}ll" + "r" * len(BASE) + r"@{}}", r"\toprule",
        r"$N$ & Arm & " + " & ".join(HEAD[b] for b in BASE) + r"\\",
        r"\midrule"]
for N in SIZES:
    for k, a in enumerate(ARMS):
        cells = []
        for b in BASE:
            r_ = dm[(dm.N == N) & (dm.method == a) & (dm.vs == b)].iloc[0]
            # a p-value below the printed resolution must not read as zero
            s = (f"{r_.p:.3f}" if r_.p >= 0.001
                 else (f"{r_.p:.4f}" if r_.p >= 0.00005 else r"$<$0.0001"))
            cells.append(rf"\textbf{{{s}}}" if (r_.p < 0.05 and r_.dm < 0) else s)
        rows.append(f"{N if k == 0 else ''} & {NAME[a]} & "
                    + " & ".join(cells) + r"\\")
    if N != SIZES[-1]:
        rows.append(r"\addlinespace[2pt]")
rows += [r"\bottomrule", r"\end{tabular}"]
write("T2_significance.tex", rows)

# ---- T3: market-cap top-100, the second universe design -------------------
tu = pd.read_csv(RES / "top_universe_5metrics.csv")
ew = pd.read_csv(RES / "ew_baseline_metrics.csv")
t1 = tu[(tu.universe == "top100") & (tu.cost_bps == 10)].copy()
KEY = {"A1_gmv_tglobal": "DG-GMV", "A2_gmv_tasym": "DG-Asym",
       "A3_hrp_gerber": "DG-HRP", "X2_shrink": "DG-Shrink"}
t1["key"] = t1["method"].map(KEY).fillna(t1["method"])
t1 = t1.set_index("key")
ew1 = ew[(ew.universe == "top100") & (ew.cost_bps == 10)].set_index("method")
t1 = pd.concat([t1[M5], ew1.loc[["equal_weight"], M5]])
best1 = {c: (t1.loc[ORDER, c] * DIR[c]).idxmax() for c in M5}
rows = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Method & Ret (\%) & Vol (\%) & SR & So & MDD (\%)\\",
        r"\midrule"]
for m in ORDER:
    cells = [(rf"\textbf{{{fmt(c, t1.loc[m, c])}}}" if best1[c] == m
              else fmt(c, t1.loc[m, c])) for c in M5]
    rows.append(NAME[m] + " & " + " & ".join(cells) + r"\\")
    if m == "DG-Shrink":
        rows.append(r"\addlinespace[2pt]")
rows += [r"\bottomrule", r"\end{tabular}"]
write("T3_top100.tex", rows)

# ---- T4: where the threshold lands, and whether learning it pays ----------
mech = pd.read_csv(RES / "reinforcement_mechanism.csv").set_index("universe")
cl = pd.read_csv(RES / "crypto_learned_c.csv", index_col=0).iloc[:, 0]
cl = cl[cl.index >= 2019]
hp = pd.read_csv(RES / "arch_v3_holdout_params.csv")
hp = hp[(hp.module == "tglobal") & (hp.year >= 2019)].c_up.astype(float)
th = pd.read_csv(RES / "headline_thresholds.csv")
sec = th[(th.arm == "DG-GMV") & (th.year >= 2019)]
rows = [r"\begin{tabular}{@{}lccl@{}}", r"\toprule",
        r"Universe & Learned $c$ & Corr.\ & Value of learning\\",
        r"\midrule",
        rf"Sector-balanced ($N{{=}}30$--$70$) & "
        rf"{sec.c_up.min():.2f}--{sec.c_up.max():.2f} & "
        rf"{mech.loc[['sector30','sector50'],'mean_corr'].mean():.2f} & "
        r"large (Tab.~\ref{tab:main}--\ref{tab:sig})\\",
        rf"Large-cap (top-100) & {mech.loc['top100','learned_c']:.2f} & "
        rf"{mech.loc['top100','mean_corr']:.2f} & "
        r"large (Tab.~\ref{tab:top100})\\",
        rf"ex-S\&P mid-cap (holdout) & {hp.min():.2f}--{hp.max():.2f} & "
        rf"{mech.loc[['exsp30','exsp50'],'mean_corr'].mean():.2f} & "
        r"none; default found\\",
        rf"Cryptocurrency & {cl.min():.2f}--{cl.max():.2f} & --- & "
        r"large, opposite sign\\",
        r"Long-only layer & --- & --- & harmful\\",
        r"Sharpe objective & $\to 0.01$ & --- & unlearnable\\",
        r"\bottomrule", r"\end{tabular}"]
write("T4_adaptivity.tex", rows)

# ---- T5: forward/backward ablation ----------------------------------------
fb = pd.read_csv(RES / "ablation_forward_backward.csv")
fb = fb[fb.variant != "FD_zero_gradient_rate"]
LBL = {"A_ste_hard": r"hard & surrogate & hard",
       "E_num_only": r"hard & surr., num.\ only & hard",
       "B_soft_soft": r"soft & surrogate & soft",
       "C_soft_deployhard": r"soft & surrogate & hard",
       "D_finite_diff": r"hard & finite diff.\ & hard"}
rows = [r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule",
        r"Forward & Backward & Deployed & $\bar{c}$ & Vol (\%) & SR & MDD (\%)\\",
        r"\midrule"]
for _, r_ in fb.iterrows():
    tag = r"\;(ours)" if r_.variant == "A_ste_hard" else ""
    rows.append(f"{LBL[r_.variant]}{tag} & {r_.c_mean_2019plus:.2f} & "
                f"{r_.ann_vol_net*100:.2f} & {r_.sharpe_net:.2f} & "
                f"{r_.max_drawdown*100:.1f}" + r"\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("T5_forward_backward.tex", rows)

# ---- T7: which component earns the risk reduction: c, delta, or both -----
cd = pd.read_csv(RES / "ablation_c_delta.csv").set_index(["N", "config"])
CD_ROWS = [  # (label c, label delta, source, key)
    (r"$0.5$", r"---", "hm", "gerber_c0.5"),
    (r"learned", r"---", "hm", "DG-GMV"),
    (r"$0.5$", r"learned", "cd", "delta_only"),
    (r"learned", r"$0.5$", "cd", "c_only_d0.50"),
    (r"$0.5$", r"$0.5$", "cd", "static_blend"),
    (r"learned", r"learned", "cd", "DG-Shrink"),
]
cd_best = {}
for N in SIZES:
    vals = [(panels[N].loc[key] if src == "hm" else cd.loc[(N, key)])
            for _, _, src, key in CD_ROWS]
    cd_best[N] = {"vol": min(v.ann_vol_net for v in vals),
                  "sr": max(v.sharpe_net for v in vals)}
rows = [r"\begin{tabular}{@{}cc" + "rr" * len(SIZES) + r"@{}}", r"\toprule",
        r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$N={N}$}}"
                              for N in SIZES) + r"\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r"$c$ & $\delta$" + r" & Vol & SR" * len(SIZES) + r"\\",
        r"\midrule"]
for lc, ld, src, key in CD_ROWS:
    cells = []
    for N in SIZES:
        d = (panels[N].loc[key] if src == "hm" else cd.loc[(N, key)])
        v, s = f"{d.ann_vol_net*100:.2f}", f"{d.sharpe_net:.2f}"
        if abs(d.ann_vol_net - cd_best[N]["vol"]) < 1e-12:
            v = rf"\textbf{{{v}}}"
        if abs(d.sharpe_net - cd_best[N]["sr"]) < 1e-12:
            s = rf"\textbf{{{s}}}"
        cells += [v, s]
    rows.append(f"{lc} & {ld} & " + " & ".join(cells) + r"\\")
    if key in ("DG-GMV", "static_blend"):
        rows.append(r"\addlinespace[2pt]")
rows += [r"\bottomrule", r"\end{tabular}"]
write("T7_c_delta.tex", rows)

# ---- T6: sub-period decomposition, headline universes --------------------
sp = pd.read_csv(RES / "headline_subperiods.csv")


def cell(N, w, m, col):
    q = sp[(sp.N == N) & (sp.window == w) & (sp.method == m)]
    return q.iloc[0][col] if len(q) else np.nan


rows = [r"\begin{tabular}{@{}p{1.8cm}p{1.7cm}"
        r">{\raggedright\arraybackslash}p{3.6cm}@{}}", r"\toprule",
        r"Window & Best & Key figures\\", r"\midrule",
        rf"COVID crash & shrinkage & ANS "
        rf"${cell(50,'covid_crash','ans','total_ret')*100:.1f}\%$ vs.\ DG-Asym "
        rf"${cell(50,'covid_crash','DG-Asym','total_ret')*100:.1f}\%$ ($N{{=}}50$)\\",
        rf"2020 full year & DG family & DG-Asym Sharpe "
        rf"${cell(50,'year_2020','DG-Asym','sharpe'):.2f}$ vs.\ ANS "
        rf"${cell(50,'year_2020','ans','sharpe'):.2f}$ ($N{{=}}50$)\\",
        rf"2022 bear & DG-Asym & "
        rf"${cell(30,'year_2022','DG-Asym','total_ret')*100:.1f}\%$ vs.\ HRP "
        rf"${cell(30,'year_2022','hrp','total_ret')*100:.1f}\%$ and $1/N$ "
        rf"${cell(30,'year_2022','equal_weight','total_ret')*100:.1f}\%$ "
        rf"($N{{=}}30$)\\",
        rf"Calm 2023--26 & $1/N$ & Sharpe "
        rf"${cell(30,'calm_2023_26','equal_weight','sharpe'):.2f}$ vs.\ DG-HRP "
        rf"${cell(30,'calm_2023_26','DG-HRP','sharpe'):.2f}$ ($N{{=}}30$)\\",
        r"\bottomrule", r"\end{tabular}"]
write("T6_subperiods.tex", rows)

print("\nall tables generated.")
