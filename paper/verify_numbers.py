"""Re-check every number in main.tex against experiments/results/.

The manuscript's headline evidence is the sector-balanced sweep at
N in {30, 70} (headline_*.csv) together with the mixed sector-plus-ETF
universes (mixed_group_*.csv); the market-cap top-100 universe carries the
methodological comparisons.  Result files for the sizes the paper does not
report (N in {20, 40, 50, 100}) stay in the release and are simply not read
here. Ordinal claims ("lowest volatility of any
method", "best in two of the three") are re-derived here rather than
spot-checked, so a rerun that reorders the panel fails loudly.

Exits non-zero on any mismatch. Run after make_tables.py.
"""

import json
import math
import pathlib
import re

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
HERE = pathlib.Path(__file__).resolve().parent
# The manuscript source is not part of the code release, so the
# hygiene section at the bottom runs only when main.tex sits next to us.
_tex, _bib = HERE / "main.tex", HERE / "refs.bib"
TEX = _tex.read_text() if _tex.exists() else ""

SIZES = [30, 70]
ARMS = ["DG-GMV", "DG-Asym", "DG-HRP", "DG-Shrink"]
BASE = ["gerber_c0.5", "ledoit_wolf", "ans", "hrp", "equal_weight"]
ok = bad = 0


def check(label, claimed, actual, tol=0.006):
    """Compare a number printed in the paper against the one in the data."""
    global ok, bad
    hit = abs(claimed - actual) <= tol
    ok, bad = ok + hit, bad + (not hit)
    print(f"  [{'OK ' if hit else 'FAIL'}] {label:52s} "
          f"paper={claimed:<10.5g} data={actual:.5g}")


def claim(label, holds):
    """Assert a qualitative claim the paper makes about an ordering."""
    global ok, bad
    ok, bad = ok + bool(holds), bad + (not holds)
    print(f"  [{'OK ' if holds else 'FAIL'}] {label}")


hm = pd.read_csv(RES / "headline_metrics.csv")
h = hm[hm.cost_bps == 10].set_index(["N", "method"])
dm = pd.read_csv(RES / "headline_dm.csv")
dm = dm[dm.N.isin(SIZES)]
panel = {N: h.loc[N].loc[ARMS + BASE] for N in SIZES}

print("== Table 1: the printed panel is complete ==")
claim("all 9 methods x 2 universes present at 10bp",
      all(len(panel[N]) == 9 and panel[N].notna().all().all() for N in SIZES))

print("\n== 6.1  what learning the threshold buys ==")
for N, margin in zip(SIZES, (0.08, 0.05)):
    check(f"N={N}: learned minus fixed default, Sharpe", margin,
          panel[N].loc["DG-GMV", "sharpe_net"]
          - panel[N].loc["gerber_c0.5", "sharpe_net"], tol=0.005)
claim("learned beats the fixed default at both sizes",
      all(panel[N].loc["DG-GMV", "sharpe_net"]
          > panel[N].loc["gerber_c0.5", "sharpe_net"] for N in SIZES))
claim("learned beats linear and nonlinear shrinkage at both sizes",
      all(panel[N].loc["DG-GMV", "sharpe_net"] > panel[N].loc[b, "sharpe_net"]
          for N in SIZES for b in ("ledoit_wolf", "ans")))
claim("DG-GMV's Sharpe margin over shrinkage is positive in all four cells",
      all(panel[N].loc["DG-GMV", "sharpe_net"] - panel[N].loc[b_, "sharpe_net"] > 0
          for N in SIZES for b_ in ("ledoit_wolf", "ans")))

for N, v in zip(SIZES, (14.26, 13.64)):
    check(f"N={N}: DG-Shrink annualized volatility (%)", v,
          panel[N].loc["DG-Shrink", "ann_vol_net"] * 100)
claim("DG-Shrink has the lowest volatility of any method, both sizes",
      all(panel[N].ann_vol_net.idxmin() == "DG-Shrink" for N in SIZES))
claim("DG-Shrink has the shallowest drawdown at N=70",
      panel[70].max_drawdown.idxmax() == "DG-Shrink")
_p_sh70 = dm[(dm.method == "DG-Shrink") & (dm.vs == "ans")
             & (dm.N == 70)].p.iloc[0]
claim(f"DG-Shrink beats ANS at the 1% level at N=70 (p={_p_sh70:.5f})",
      _p_sh70 < 0.01)
_p_sh30 = dm[(dm.method == "DG-Shrink") & dm.vs.isin(["ledoit_wolf", "ans"])
             & (dm.N == 30)].p.min()
claim(f"...and the paper does NOT claim significance at N=30 "
      f"(smallest p there is {_p_sh30:.3f})",
      _p_sh30 > 0.05 and (not TEX or "not significant" in TEX))

claim("DG-HRP has the best Sharpe and Sortino of any method at N=30 and N=70",
      all(panel[N].sharpe_net.idxmax() == "DG-HRP"
          and panel[N].sortino_net.idxmax() == "DG-HRP" for N in (30, 70)))
check("DG-HRP beats sample-correlation HRP in how many universes", 2,
      sum(panel[N].loc["DG-HRP", "sharpe_net"]
          > panel[N].loc["hrp", "sharpe_net"] for N in SIZES), tol=0)

for N, v in zip(SIZES, (12.52, 10.36)):
    check(f"N={N}: 1/N annualized return (%)", v,
          panel[N].loc["equal_weight", "ann_ret_net"] * 100)
claim("1/N earns the highest raw return at both sizes",
      all(panel[N].ann_ret_net.idxmax() == "equal_weight" for N in SIZES))
claim("1/N has the highest volatility and deepest drawdown at both sizes",
      all(panel[N].ann_vol_net.idxmax() == "equal_weight"
          and panel[N].max_drawdown.idxmin() == "equal_weight" for N in SIZES))
_pew = dm[dm.vs == "equal_weight"].p
claim(f"all 8 cells beat 1/N on realized variance at p <= 0.0054 "
      f"(largest is {_pew.max():.5f})",
      len(_pew) == 8 and _pew.max() <= 0.0054)
claim("...and every one of those tests points our way",
      (dm[dm.vs == "equal_weight"].dm < 0).all())

print("\n== 6.1  top-100, the second design ==")
tu = pd.read_csv(RES / "top_universe_5metrics.csv")
t100 = tu[(tu.universe == "top100") & (tu.cost_bps == 10)].set_index("method")
DIRS = {"ann_ret_net": 1, "ann_vol_net": -1, "sharpe_net": 1,
        "sortino_net": 1, "max_drawdown": 1, "avg_turnover": -1}
claim("DG-HRP improves on plain HRP in every column of Table 3",
      all(d * (t100.loc["A3_hrp_gerber", c] - t100.loc["hrp", c]) > 0
          for c, d in DIRS.items()))
d100 = pd.read_csv(RES / "main_dm_gs2022.csv")
check("learned vs linear shrinkage, strongest single p-value", 0.0004,
      d100[(d100.method == "diffgerber_tglobal")
           & (d100.vs == "ledoit_wolf")].p.iloc[0], tol=0.00006)
est = dm[(dm.vs != "equal_weight") & (dm.dm < 0)]          # in our favour only
# the superlative is scoped to the equity designs: the mixed universes reach
# p < 1e-5 against Ledoit-Wolf, which is stronger, so the claim must not be
# stated over the whole study
_p_top = d100[(d100.method == "diffgerber_tglobal")
              & (d100.vs == "ledoit_wolf")].p.iloc[0]
claim("...and it is our smallest p against a rival estimator in the equity "
      "designs",
      _p_top < min(est.p.min(),
                   pd.read_csv(RES / "arch_v3_holdout_dm.csv").p.min()))
_gmix = pd.concat([pd.read_csv(RES / "mixed_group_dm.csv"),
                   pd.read_csv(RES / "mixed_arms_dm.csv")])
_gmix = _gmix[_gmix.universe.isin(["mix30", "mix70"])
              & _gmix.vs.isin(["ledoit_wolf", "ans", "hrp"]) & (_gmix.dm < 0)]
claim(f"...but NOT over the whole study, since the mixed design reaches "
      f"p={_gmix.p.min():.5f}; the text says 'among the equity designs'",
      _gmix.p.min() < _p_top
      and (not TEX
           or "the strongest such comparison among the equity designs" in TEX))
claim("...though the tests against 1/N are sharper still",
      dm[dm.vs == "equal_weight"].p.min() < 0.0004)
b1 = pd.read_csv(RES / "reinforcement_boot.csv").query(
    "universe == 'top100'").set_index("stat")
for st, obs, pv in (("dSharpe", 0.029, 0.069), ("dSortino", 0.040, 0.074)):
    check(f"stationary bootstrap {st}", obs, b1.loc[st, "observed"], tol=0.0006)
    check(f"stationary bootstrap {st}, one-sided p", pv,
          b1.loc[st, "p_onesided"], tol=0.0006)

st = pd.read_csv(RES / "ablation_stability_gs2022_top100.csv").set_index("label")

print("\n== 6.1  which component earns the risk reduction ==")
cdl = pd.read_csv(RES / "ablation_c_delta.csv").set_index(["N", "config"])
cdm = pd.read_csv(RES / "ablation_c_delta_dm.csv")
_FACT = {"c_only_d0.50": ("learning only c, blend fixed at 0.5",
                          {30: 14.21, 70: 13.82}),
         "delta_only":   ("learning only the blend", {30: 14.36, 70: 13.75}),
         "DG-Shrink":    ("joint", {30: 14.26, 70: 13.64})}
for _cfg, (_lbl, _want) in _FACT.items():
    for _N in SIZES:
        check(f"N={_N}: {_lbl}, volatility (%)", _want[_N],
              cdl.loc[(_N, _cfg), "ann_vol_net"] * 100)
claim("joint learning beats both one-component variants at N=70, "
      "which is the only place the paper claims it",
      cdl.loc[(70, "DG-Shrink"), "ann_vol_net"]
      < min(cdl.loc[(70, "c_only_d0.50"), "ann_vol_net"],
            cdl.loc[(70, "delta_only"), "ann_vol_net"])
      and cdl.loc[(30, "DG-Shrink"), "ann_vol_net"]
      > cdl.loc[(30, "c_only_d0.50"), "ann_vol_net"])
claim("delta learned alone never beats the static blend",
      all(cdl.loc[(N, "delta_only"), "ann_vol_net"]
          >= cdl.loc[(N, "static_blend"), "ann_vol_net"] - 1e-9
          for N in SIZES))
claim("no within-family difference is significant at the 1% level "
      f"(smallest p = {cdm.p.min():.3f})", cdm.p.min() > 0.01)

print("\n== 6.2  the learned threshold recovers the oracle ==")
g = pd.read_csv(RES / "threshold_grid_gs2022.csv").set_index("threshold")
for lbl, key, v in (("published default c=0.5", "c=0.5", 14.04),
                    ("grid-plus-validation", "val_select", 13.92),
                    ("learned", "learned", 13.76),
                    ("ex-post oracle c=1.0", "c=1.0", 13.72)):
    check(f"top-100 volatility (%), {lbl}", v, g.loc[key, "ann_vol_net"] * 100)
claim("the ex-post best fixed threshold on the grid is c=1.0",
      g[g.index.str.startswith("c=")].ann_vol_net.idxmin() == "c=1.0")
x3 = pd.read_csv(RES / "reinforcement_x3.csv")
x3y = x3[x3.year != "METRICS"].astype(
    {"c_up": float, "c_down": float, "delta": float})
for lbl, col, v in (("c_up", "c_up", 1.6), ("c_down", "c_down", 1.5),
                    ("delta", "delta", 0.75)):
    check(f"joint three-parameter run, terminal {lbl}", v,
          x3y[col].iloc[-1], tol=0.05)
check("joint three-parameter volatility (%)", 13.70,
      float(x3[x3.year == "METRICS"].iloc[0].ann_vol_net) * 100)
ag = pd.read_csv(RES / "reinforcement_asym_grid.csv")
agg = ag[(ag.universe == "top100") & (ag.kind == "grid")]
check("size of the two-dimensional threshold grid", 25, len(agg), tol=0)
claim("the grid optimum is the symmetric point (1.0, 1.0)",
      tuple(agg.loc[agg.ann_vol_net.idxmin(), ["c_up", "c_down"]]) == (1.0, 1.0))
check("grid optimum volatility (%)", 13.72, agg.ann_vol_net.min() * 100)

print("\n== 3.2  how far the relaxation sits from the statistic ==")
rg = pd.read_csv(RES / "relaxation_gap_top100.csv").set_index("tau")
check("evaluation folds behind the gap measurement", 114,
      int(rg.loc[0.02, "n_folds"]), tol=0)
check("correlation gap at tau=0.02 (%, Frobenius)", 6.0,
      rg.loc[0.02, "corr_gap_frobenius_mean"] * 100, tol=0.3)
check("decision gap at tau=0.02 (% of gross notional)", 14.0,
      rg.loc[0.02, "weight_gap_l1_mean"] * 100, tol=0.5)
check("decision gap at tau=0.2, where annealing begins (%)", 84.0,
      rg.loc[0.2, "weight_gap_l1_mean"] * 100, tol=0.5)
claim("the decision gap exceeds the matrix gap at every temperature",
      (rg.weight_gap_l1_mean > rg.corr_gap_frobenius_mean).all())

print("\n== 3.3  gradient flow through the statistic ==")
gd = pd.read_csv(RES / "gradient_decomposition.csv")
claim("decomposition measured on the paper's 114 evaluation folds",
      (gd.n_folds == 114).all())
hi_t = gd[gd.tau == 0.2].den_share_mean
lo_t = gd[gd.tau == 0.02].den_share_mean
check("denominator share at tau=0.2, low (%)", 51, hi_t.min() * 100, tol=0.6)
check("denominator share at tau=0.2, high (%)", 62, hi_t.max() * 100, tol=0.6)
claim(f"about a third of the gradient at tau=0.02 "
      f"({lo_t.min()*100:.0f}-{lo_t.max()*100:.0f}%)",
      0.28 <= lo_t.min() and lo_t.max() <= 0.37)
claim("the numerator-only aggregate step never flips sign",
      (gd.aggregate_sign_flip == 0).all())
check("channels oppose, low (% of folds)", 36,
      gd.opposing_sign_frac.min() * 100, tol=0.6)
check("channels oppose, high (% of folds)", 41,
      gd.opposing_sign_frac.max() * 100, tol=0.6)
check("per-fold sign flips if the denominator is dropped, high (%)", 25,
      gd.fold_sign_flip_frac.max() * 100, tol=0.6)
claim("the manuscript now reports the decomposition, not just its "
      "consequence",
      not TEX
      or ("51 to 62 percent of the gradient magnitude" in TEX
          and "36 to 41 percent of folds" in TEX
          and "though the aggregated step keeps its sign" in TEX))

print("\n== 6.3  does the answer depend on the relaxation? ==")
fb = pd.read_csv(RES / "ablation_forward_backward.csv")
z = fb[fb.variant == "FD_zero_gradient_rate"].iloc[0]
fbi = fb[fb.variant != "FD_zero_gradient_rate"].set_index("variant")
check("straight-through mean threshold, 2019+", 1.09,
      fbi.loc["A_ste_hard", "c_mean_2019plus"], tol=0.005)
check("finite-difference mean threshold, 2019+", 0.99,
      fbi.loc["D_finite_diff", "c_mean_2019plus"], tol=0.005)
check("straight-through volatility (%)", 13.76,
      fbi.loc["A_ste_hard", "ann_vol_net"] * 100)
check("finite-difference volatility (%)", 13.72,
      fbi.loc["D_finite_diff", "ann_vol_net"] * 100)
check("soft-forward drift in the threshold", 1.46,
      fbi.loc["B_soft_soft", "c_mean_2019plus"], tol=0.005)
check("numerator-only threshold drifts to", 1.40,
      fbi.loc["E_num_only", "c_mean_2019plus"], tol=0.005)
check("...costing this much volatility (bp)", 25,
      (fbi.loc["E_num_only", "ann_vol_net"]
       - fbi.loc["A_ste_hard", "ann_vol_net"]) * 1e4, tol=0.6)
claim("the numerator-only threshold overshoots the oracle from above",
      fbi.loc["E_num_only", "c_mean_2019plus"] > 1.0
      > abs(fbi.loc["A_ste_hard", "c_mean_2019plus"]
            - fbi.loc["E_num_only", "c_mean_2019plus"]))
claim("the two mutilated gradients are the two worst rows on volatility",
      set(fbi.ann_vol_net.nlargest(2).index)
      == {"C_soft_deployhard", "E_num_only"})
claim("the finite difference is marginally better on volatility and drawdown",
      fbi.loc["D_finite_diff", "ann_vol_net"] < fbi.loc["A_ste_hard", "ann_vol_net"]
      and fbi.loc["D_finite_diff", "max_drawdown"]
      > fbi.loc["A_ste_hard", "max_drawdown"])
check("finite-difference gradient was zero at how many steps", 0.0,
      z.c_mean_2019plus, tol=1e-9)
check("finite-difference steps taken", 400, z.c_final, tol=0)
# Table 5's caption claims row 1 is an independent retrain that reproduces
# the headline threshold path; check that, since it is a reproducibility claim
fbt = pd.read_csv(RES / "ablation_fb_thresholds.csv")
retrain = fbt[fbt.variant == "A_ste_hard"].set_index("year").c
head = st.loc["tglobal_init0.5", [f"c_{y}" for y in range(2017, 2027)]]
head.index = range(2017, 2027)
diff = (retrain - head).abs()
check("retrain reproduces the headline threshold in how many of ten years", 8,
      int((diff < 1e-9).sum()), tol=0)
check("largest disagreement in the other two", 0.007, float(diff.max()), tol=0.0006)
check("...worth this much annualized volatility (bp)", 0.5,
      abs(fbi.loc["A_ste_hard", "ann_vol_net"]
          - g.loc["learned", "ann_vol_net"]) * 1e4, tol=0.2)

tm = pd.read_csv(RES / "ablation_fb_timing.csv").set_index("variant")
check("finite difference, mean seconds per refit", 5.2,
      tm.loc["D_finite_diff", "mean_seconds_per_refit"], tol=0.06)
check("surrogate, mean seconds per refit", 9.4,
      tm.loc["A_ste", "mean_seconds_per_refit"], tol=0.06)
claim("the finite difference really is the cheaper of the two here",
      tm.loc["D_finite_diff", "mean_seconds_per_refit"]
      < tm.loc["A_ste", "mean_seconds_per_refit"])

print("\n== 6.4  what the threshold turns out to be ==")
cl = pd.read_csv(RES / "crypto_learned_c.csv", index_col=0).iloc[:, 0]
cl = cl[cl.index >= 2019]
check("crypto learned threshold, low", 0.15, cl.min(), tol=0.005)
check("crypto learned threshold, high", 0.36, cl.max(), tol=0.005)
cr = pd.read_csv(RES / "crypto_results.csv").set_index("method")
check("crypto volatility (%) at c=0.5", 67.8,
      cr.loc["gerber_c0.5", "ann_vol_net"] * 100, tol=0.06)
check("crypto volatility (%) at c=1.5", 70.2,
      cr.loc["gerber_c1.5", "ann_vol_net"] * 100, tol=0.06)
_seq = list(cr.loc[["gerber_c0.5", "gerber_c1.0", "gerber_c1.5"], "ann_vol_net"])
claim("raising the crypto threshold hurts monotonically", _seq == sorted(_seq))
check("cost of transferring an equity threshold to crypto (bp)", 240,
      (cr.loc["gerber_c1.5", "ann_vol_net"]
       - cr.loc["gerber_c0.5", "ann_vol_net"]) * 1e4, tol=6)

sp_ = lambda x: math.log1p(math.exp(x))
j = json.loads((RES / "scalar_thresholds_gs2022.json").read_text())
check("asymmetric threshold, upside (2026)", 0.99,
      sp_(j["tasym_2026"]["raw_up"]), tol=0.01)
check("asymmetric threshold, downside (2026)", 1.32,
      sp_(j["tasym_2026"]["raw_down"]), tol=0.01)
pl = ag[(ag.universe == "top100")
        & (ag.kind == "expost_best_vs_learned")].iloc[0]
check("learned asymmetric path vs 2-D grid oracle, p", 0.011,
      pl.sortino_net, tol=0.001)

sp = pd.read_csv(RES / "headline_subperiods.csv")
q = lambda N, w, m, c: sp[(sp.N == N) & (sp.window == w)
                          & (sp.method == m)].iloc[0][c]
for m, v in (("DG-Asym", -6.2), ("hrp", -16.7), ("equal_weight", -23.4)):
    check(f"2022 bear at N=30, {m} total return (%)", v,
          q(30, "year_2022", m, "total_ret") * 100, tol=0.06)
claim("DG-Asym is the best method of 2022 at N=30",
      sp[(sp.N == 30) & (sp.window == "year_2022")]
      .set_index("method").total_ret.idxmax() == "DG-Asym")

dd = pd.read_csv(RES / "decision_dependence_thresholds.csv")
check("long-only threshold path, low", 0.66, dd.c_longonly.min(), tol=0.01)
check("long-only threshold path, high", 1.38, dd.c_longonly.max(), tol=0.01)

fr = pd.read_csv(RES / "reinforcement_frobenius.csv")
fy = fr[fr.year != "METRICS"].astype({"c_frobenius": float, "c_decision": float})
check("Frobenius-trained threshold, low", 0.46, fy.c_frobenius.min(), tol=0.01)
check("Frobenius-trained threshold, high", 0.51, fy.c_frobenius.max(), tol=0.01)
check("decision-trained threshold, low", 0.89, fy.c_decision.min(), tol=0.01)
check("decision-trained threshold, high", 1.23, fy.c_decision.max(), tol=0.01)
frm = fr[fr.year == "METRICS"].iloc[0]
check("Frobenius-trained volatility (%)", 14.08, float(frm.ann_vol_net) * 100)
check("Frobenius-trained Sharpe", 0.44, float(frm.sharpe_net), tol=0.006)
check("decision-trained volatility (%)", 13.76, g.loc["learned", "ann_vol_net"] * 100)
check("decision-trained Sharpe", 0.54, g.loc["learned", "sharpe_net"], tol=0.006)

ad = pd.read_csv(RES / "reinforcement_analytic_delta.csv").iloc[0]
x2 = pd.read_csv(RES / "top_universe_x2_params.csv")
x2t = x2[x2.universe == "top100"]
shr100 = t100.loc["X2_shrink"]
check("formula shrinkage intensity", 0.27, ad.delta_mean, tol=0.006)
check("learned shrinkage intensity at top-100, low", 0.68, x2t.delta.min(), tol=0.006)
check("learned shrinkage intensity at top-100, high", 0.86, x2t.delta.max(), tol=0.006)
claim("the formula loses on volatility, Sharpe and drawdown at once",
      ad.ann_vol_net > shr100.ann_vol_net
      and ad.sharpe_net < shr100.sharpe_net
      and ad.max_drawdown < shr100.max_drawdown)
ht = pd.read_csv(RES / "headline_thresholds.csv")
alld = pd.concat([x2.delta, ht[(ht.arm == "DG-Shrink") & ht.delta.notna()].delta])
check("shrinkage intensity across equity universes, low", 0.47, alld.min(), tol=0.006)
check("shrinkage intensity across equity universes, high", 0.87, alld.max(), tol=0.006)
check("shrinkage intensity across equity universes, median", 0.67,
      alld.median(), tol=0.006)

# GUARD, not a paper claim. Sector N=100 is not reported in the manuscript,
# but it is why the mechanism paragraph says the map locates the optimum
# rather than predicting the gain: at N=100 the ambient correlation and the
# ex-post optimum both match N=30/50, yet the annual refit lands behind the
# published default. Kept here so the stronger claim cannot creep back in.
tc = pd.read_csv(RES / "threshold_ceiling.csv")
for N in sorted(tc.N.unique()):
    s = tc[tc.N == N]
    grid = s[s.rule.str.startswith("fixed")]
    best = grid.loc[grid.sharpe_net.idxmax()]
    d0 = grid[grid.c == 0.5].iloc[0]
    captured = s[s.rule == "learned"].iloc[0].sharpe_net - d0.sharpe_net
    check(f"sector N={N}: ex-post optimal threshold", 1.5, best.c, tol=0.001)
    claim(f"sector N={N}: the optimum sits well above the default "
          f"(headroom {best.sharpe_net - d0.sharpe_net:+.3f})",
          best.sharpe_net - d0.sharpe_net > 0.09)
    if N <= 70:
        claim(f"sector N={N}: and the refit captures part of it "
              f"({captured:+.3f})", captured > 0)
    else:
        claim(f"sector N={N}: but the refit does NOT capture it "
              f"({captured:+.3f}) -- the boundary behind the scoped claim",
              captured < 0)

me = pd.read_csv(RES / "reinforcement_mechanism.csv")
check("threshold vs mean pairwise correlation, rho", 0.96,
      float(np.corrcoef(me.mean_corr, me.learned_c)[0, 1]), tol=0.006)
check("number of equity universes behind the mechanism claim", 6, len(me), tol=0)
hi, lo = me[me.mean_corr > 0.25], me[me.mean_corr < 0.20]
check("high-correlation universes: mean correlation", 0.3, hi.mean_corr.mean(), tol=0.04)
check("high-correlation universes: threshold, low", 1.0, hi.learned_c.min(), tol=0.08)
check("high-correlation universes: threshold, high", 1.2, hi.learned_c.max(), tol=0.08)
check("low-correlation universes: mean correlation", 0.15, lo.mean_corr.mean(), tol=0.02)
check("low-correlation universes: threshold stays at the default", 0.5,
      lo.learned_c.mean(), tol=0.02)
# The paper reads the map as two levels, not a slope. That is a claim about
# the shape of these six points, so check the shape and not just rho.
check("high-correlation group size", 4, len(hi), tol=0)
check("low-correlation group size", 2, len(lo), tol=0)
claim("the two groups are cleanly separated in correlation",
      lo.mean_corr.max() < hi.mean_corr.min() - 0.1)
claim("...and in the threshold they imply",
      lo.learned_c.max() < hi.learned_c.min() - 0.5)
_rin = float(np.corrcoef(hi.mean_corr, hi.learned_c)[0, 1])
claim(f"no upward gradient inside the high group (rho = {_rin:+.2f})", _rin <= 0)

print("\n== 6.6  when the original default is the right choice ==")
sth = pd.read_csv(RES / "sharpe_objective_thresholds_top100.csv")
check("Sharpe objective collapses the threshold to", 0.012,
      float(sth.c_sharpe_trained.iloc[-1]), tol=0.001)
claim("...monotonically, from a start near the default",
      sth.c_sharpe_trained.iloc[-1] < sth.c_sharpe_trained.iloc[0] < 1.0)
sh = pd.read_csv(RES / "sharpe_objective_top100.csv").set_index("method")
check("Sharpe-trained out-of-sample Sharpe", 0.30,
      sh.loc["sharpe_trained", "sharpe_net"], tol=0.006)
check("variance-trained out-of-sample Sharpe", 0.54,
      sh.loc["variance_trained", "sharpe_net"], tol=0.006)
claim("the Sharpe-trained threshold loses on its own metric",
      sh.loc["sharpe_trained", "sharpe_net"]
      < sh.loc["variance_trained", "sharpe_net"])

tstate = st[st.index.str.startswith("tstate")]
tglobal = st[st.index.str.startswith("tglobal")]
check("state-conditioned network volatility (%), low", 14.04,
      tstate.ann_vol_net.min() * 100)
check("state-conditioned network volatility (%), high", 14.44,
      tstate.ann_vol_net.max() * 100)
claim("the single scalar beats the state-conditioned network under every seed",
      tglobal.ann_vol_net.max() < tstate.ann_vol_net.min())
_ts = d100[(d100.method == "diffgerber_tstate")
           & (d100.vs == "gerber_gs2022_c0.5")].iloc[0]
check("state-conditioned network vs fixed default, DM statistic", 1.97,
      _ts.dm, tol=0.006)
check("state-conditioned network vs fixed default, p", 0.051, _ts.p, tol=0.0006)
claim("...and the sign says the network is the worse of the two", _ts.dm > 0)

mcap = pd.read_csv(RES / "main_metrics_gs2022.csv").set_index("method")
ladder = ["diffgerber_tglobal", "diffgerber_tasym",
          "diffgerber_tasset", "diffgerber_tstate"]
vols = list(mcap.loc[ladder, "ann_vol_net"])
claim("the modules degrade monotonically in parameter count on the "
      "variance they optimize", vols == sorted(vols))

ew_sec = pd.read_csv(RES / "ew_sector_metrics.csv")
# cross-check of an independent run of the same protocol; it was produced for
# N in {30, 50, 100}, so only the sizes it shares with the headline pack apply
for N in sorted(set(SIZES) & set(ew_sec.N.unique())):
    a = panel[N].sharpe_net
    b = ew_sec[(ew_sec.N == N) & (ew_sec.cost_bps == 10)].set_index("method").sharpe_net
    claim(f"ew_sector agrees with the headline pack at N={N}",
          (a - b[a.index]).abs().max() < 1e-9)

lo_ = pd.read_csv(RES / "decision_dependence.csv").set_index("method")
check("long-only, learned threshold volatility (%)", 15.00,
      lo_.loc["lo_learned", "ann_vol_net"] * 100)
check("long-only, fixed default volatility (%)", 14.84,
      lo_.loc["lo_fixed0.5", "ann_vol_net"] * 100)
claim("learning is worse than the default under a long-only layer",
      lo_.loc["lo_learned", "ann_vol_net"] > lo_.loc["lo_fixed0.5", "ann_vol_net"])
lodm = pd.read_csv(RES / "decision_dependence_dm.csv")
check("long-only, learned vs default, p", 0.023,
      lodm[lodm.vs == "lo_fixed0.5"].p.iloc[0], tol=0.0006)
claim("...and significantly so, in the wrong direction",
      lodm[lodm.vs == "lo_fixed0.5"].dm.iloc[0] > 0)

check("development ceiling, best cells won in sample", 15,
      pd.read_csv(RES / "arch_v3_winner.csv").cells_won.max(), tol=0)
hd = pd.read_csv(RES / "arch_v3_holdout_dm.csv")
check("holdout cells retained, N=30", 15,
      hd[(hd.N == 30) & (hd.vs == "CELLS")].dm.iloc[0], tol=0)
check("holdout cells retained, N=50", 13,
      hd[(hd.N == 50) & (hd.vs == "CELLS")].dm.iloc[0], tol=0)
hmet = pd.read_csv(RES / "arch_v3_holdout_metrics.csv")
hfull = hmet[(hmet.window == "full_2017_26") & (hmet.cost_bps == 10)]
check("holdout: linear shrinkage annual return (%)", -4.1,
      hfull[(hfull.N == 30)
            & (hfull.method == "ledoit_wolf")].ann_ret_net.iloc[0] * 100, tol=0.06)
for N in (30, 50):
    w = hfull[(hfull.N == N) & hfull.method.str.startswith("WINNER")].iloc[0]
    for b in ("ledoit_wolf", "ans", "hrp"):
        claim(f"holdout N={N}: winner beats {b} on Sharpe",
              w.sharpe_net
              > hfull[(hfull.N == N) & (hfull.method == b)].sharpe_net.iloc[0])
    claim(f"holdout N={N}: winner does NOT beat the fixed default",
          w.sharpe_net < hfull[(hfull.N == N)
                               & (hfull.method == "gerber_c0.5")].sharpe_net.iloc[0])
hp = pd.read_csv(RES / "arch_v3_holdout_params.csv")
check("holdout: the threshold converges to", 0.48,
      float(hp[(hp.module == "tglobal") & (hp.year >= 2024)].c_up.astype(float).mean()),
      tol=0.01)
check("holdout universes: mean pairwise correlation", 0.16,
      me[me.universe.str.startswith("exsp")].mean_corr.mean(), tol=0.01)

check("observations per asset at the top of the reported range", 3.6,
      252 / 70, tol=0.05)

print("\n== 6.7  simulation ==")
sim = pd.read_csv(RES / "simulation_contamination.csv").set_index("contamination")
r3 = sim.loc["p=0.03,size=8.0"]
check("3% contamination: Gerber excess variance", 1.174, r3.excess_at_cstar, tol=0.0006)
check("3% contamination: linear shrinkage", 1.498, r3.excess_ledoit_wolf, tol=0.0006)
check("3% contamination: nonlinear shrinkage", 1.417, r3.excess_ans, tol=0.0006)
heavy = sim.loc[["p=0.05,size=8.0", "p=0.03,size=15.0"]]
claim("under the heaviest contamination both shrinkage estimators fall "
      "behind the sample covariance",
      ((heavy.excess_ledoit_wolf > heavy.excess_sample)
       & (heavy.excess_ans > heavy.excess_sample)).all())
claim("...and the Gerber estimator stays ahead of all three there",
      (heavy.excess_at_cstar < heavy[["excess_sample", "excess_ledoit_wolf",
                                      "excess_ans"]].min(axis=1)).all())
tt = pd.read_csv(RES / "simulation_two_targets.csv").set_index("contamination")
check("clean-covariance optimum, no contamination", 0.4,
      tt.loc["p=0.0,size=0.0", "c_star_statistical"], tol=0.001)
check("clean-covariance optimum falls to", 0.2,
      tt.loc["p=0.05,size=8.0", "c_star_statistical"], tol=0.001)
check("realized-variance optimum rises to", 1.5,
      tt.loc["p=0.05,size=8.0", "c_star_decision"], tol=0.001)

print("\n== 6.8  robustness ==")
check("initialization sweep, lowest volatility (%)", 13.73, tglobal.ann_vol_net.min() * 100)
check("initialization sweep, highest volatility (%)", 13.87, tglobal.ann_vol_net.max() * 100)
lr = pd.read_csv(RES / "conditioning_control_top400.csv").query(
    "method == 'learned'").ann_vol_net * 100
check("ridge sweep, lowest volatility (%)", 12.75, lr.min())
check("ridge sweep, highest volatility (%)", 12.86, lr.max())
rk = pd.read_csv(RES / "rank_baselines.csv")
rk1 = rk[(rk.universe == "top100") & (rk.eval_from == 2017)].set_index("method")
check("learned vs rank baselines: learned (%)", 13.76,
      rk1.loc["learned", "ann_vol_net"] * 100)
check("shrunk Spearman (%)", 14.35, rk1.loc["spearman_shrunk", "ann_vol_net"] * 100)
check("shrunk Kendall (%)", 14.48, rk1.loc["kendall_shrunk", "ann_vol_net"] * 100)
claim("the unregularized rank correlations are far behind",
      min(rk1.loc["spearman", "ann_vol_net"], rk1.loc["kendall", "ann_vol_net"])
      > rk1.loc["kendall_shrunk", "ann_vol_net"])

e19 = pd.read_csv(RES / "eval_from_2019.csv")
e19t = e19[e19.universe == "top100"].set_index("method")
for meth, v in (("learned", 14.63), ("gerber_c0.5", 14.98),
                ("ans", 15.25), ("ledoit_wolf", 15.66)):
    check(f"2019+ top-100 volatility (%), {meth}", v,
          e19t.loc[meth, "ann_vol_net"] * 100)
claim("from 2019 the learned threshold beats every baseline",
      e19t.ann_vol_net.idxmin() == "learned")

print("\n== Limitations ==")
claim("DG-HRP's Sharpe and Sortino clear 1/N at both reported sizes",
      all(panel[N].loc["DG-HRP", "sharpe_net"]
          > panel[N].loc["equal_weight", "sharpe_net"]
          and panel[N].loc["DG-HRP", "sortino_net"]
          > panel[N].loc["equal_weight", "sortino_net"] for N in SIZES))
claim("1/N earns more raw return than every optimized rule, both sizes",
      all(panel[N].loc["equal_weight", "ann_ret_net"]
          > panel[N].loc[ARMS + BASE[:-1], "ann_ret_net"].max() for N in SIZES))

print("\n== table hygiene ==")
_t1 = (HERE / "tables" / "T1_headline_panel.tex").read_text()
_cells = {}
for _line in _t1.splitlines():
    if "&" not in _line or _line.startswith("\\") or "multicolumn" in _line:
        continue
    _cells[_line.split("&")[0].strip()] = [c.strip() for c in _line.split("&")[1:]]
_LBL = {"DG-GMV": "DG-GMV", "DG-Asym": "DG-Asym", "DG-HRP": "DG-HRP",
        "DG-Shrink": "DG-Shrink", "gerber_c0.5": "Gerber $c{=}0.5$",
        "ledoit_wolf": "Ledoit--Wolf", "ans": "ANS", "hrp": "HRP",
        "equal_weight": "Equal weight $1/N$"}
_wrong = []
for _i, _N in enumerate(SIZES):
    for _j, _c in enumerate(["ann_ret_net", "ann_vol_net", "sharpe_net",
                             "sortino_net", "max_drawdown"]):
        _want = (panel[_N][_c] * DIRS[_c]).idxmax()
        _got = [m for m in ARMS + BASE
                if "textbf" in _cells[_LBL[m]][_i * 5 + _j]]
        if _got != [_want]:
            _wrong.append((_N, _c, _got, _want))
claim(f"bold marks the best value in every generated column {_wrong}",
      not _wrong)
_t2 = (HERE / "tables" / "T2_significance.tex").read_text()
claim("no p-value is printed as a bare zero", "0.0000" not in _t2)

if TEX:
    print("\n== manuscript hygiene (rewritten manuscript) ==")
    labels = set(re.findall(r"\\label\{([^}]+)\}", TEX))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", TEX))
    claim(f"no dangling cross-references {sorted(refs - labels)}", not refs - labels)
    if _bib.exists():
        cited = {k.strip() for grp in re.findall(r"\\citep?\{([^}]+)\}", TEX)
                 for k in grp.split(",")}
        bib = set(re.findall(r"@\w+\{([^,]+),", _bib.read_text()))
        claim(f"every citation resolves in refs.bib {sorted(cited - bib)}",
              not cited - bib)
    for banned in ("never enters the degenerate region", "first dynamic Gerber",
                   "differentiable Gerber statistic"):
        claim(f"retracted claim absent: {banned!r}", banned.lower() not in TEX.lower())
    claim("no Korean text in the source",
          not re.search(r"[\uac00-\ud7af]", TEX))
    claim("reproducibility URL present",
          "github.com/jubro839/DiffGerber" in TEX)
    claim("AI-tools disclosure present in the body",
          "Generative AI tools" in TEX)

    print("\n== multi-asset claims ==")
    mam = pd.read_csv(RES / "multiasset_metrics.csv")
    mad = pd.read_csv(RES / "multiasset_dm.csv")
    mat = pd.read_csv(RES / "multiasset_thresholds.csv")
    check("learned threshold across ETF universes, low", 0.11,
          mat.c.min(), tol=0.005)
    check("learned threshold across ETF universes, high", 0.31,
          mat.c.max(), tol=0.005)
    for _u, _lv, _dv in (("ma9", 4.99, 5.08), ("ma12", 4.01, 4.31)):
        _s = mam[mam.universe == _u].set_index("method")
        check(f"{_u}: learned volatility (%)", _lv,
              _s.loc["tglobal", "ann_vol_net"] * 100, tol=0.006)
        check(f"{_u}: published-rule volatility (%)", _dv,
              _s.loc["gerber_c0.5", "ann_vol_net"] * 100, tol=0.006)
        _g = _s[_s.index.str.startswith("grid_")]
        claim(f"{_u}: learner finishes below the best constant grid threshold",
              _s.loc["tglobal", "ann_vol_net"] < _g.ann_vol_net.min())
        if _u == "ma9":     # at ma12, linear shrinkage edges EW on Sharpe
            claim(f"{_u}: equal weighting keeps the highest Sharpe",
                  _s.loc["equal_weight", "sharpe_net"]
                  == _s.drop(index=[i for i in _s.index
                                    if i.startswith("grid_")]).sharpe_net.max())
        else:
            claim(f"{_u}: EW Sharpe still exceeds the learner's",
                  _s.loc["equal_weight", "sharpe_net"]
                  > _s.loc["tglobal", "sharpe_net"])
        claim(f"{_u}: EW runs at more than twice the learner's volatility",
              _s.loc["equal_weight", "ann_vol_net"]
              > 2 * _s.loc["tglobal", "ann_vol_net"])
    _d9 = mad[(mad.universe == "ma9") & (mad.vs == "gerber_c0.5")].iloc[0]
    _d12 = mad[(mad.universe == "ma12") & (mad.vs == "gerber_c0.5")].iloc[0]
    check("ma9: DM vs published rule", -3.32, _d9.dm, tol=0.006)
    check("ma12: DM vs published rule", -4.74, _d12.dm, tol=0.006)
    claim(f"both differentials inside the 1% level "
          f"(p={_d9.p:.5f}, {_d12.p:.5f})", max(_d9.p, _d12.p) < 0.01)

    print("\n== Table 1 parsed out of main.tex, cell by cell ==")
    # Rather than re-typing the panel here, read the printed table back and
    # reconcile every cell with the CSV it came from.  A rerun that reorders
    # or rescales the panel therefore fails loudly.
    _tbl = TEX[TEX.index(r"\label{tabmain}"):]
    _tbl = _tbl[:_tbl.index(r"\end{table*}")]
    _iB = _tbl.index("Panel B.")

    def _cells(_row):
        """LaTeX row -> (label, [10 floats])."""
        _parts = _row.split("&")
        _lab = _parts[0].strip()
        _out = []
        for _c in _parts[1:]:
            _c = re.sub(r"\^\{?\**\}?", "", _c.replace(r"\\", ""))
            _c = re.sub(r"\\(?:text|math)bf\{([^}]*)\}", r"\1", _c)
            _c = _c.replace("$", "").strip()
            try:
                _out.append(float(_c))
            except ValueError:          # the column header row
                return _lab, None
        return _lab, _out

    def _rows(_chunk):
        for _line in _chunk.splitlines():
            _line = _line.strip()
            if ("&" not in _line or "multicolumn" in _line
                    or _line.startswith("\\")):
                continue
            _lab, _vals = _cells(_line)
            if _vals is not None and len(_vals) == 10:
                yield _lab, _vals

    _COLS = ["ann_ret_net", "ann_vol_net", "sharpe_net",
             "sortino_net", "max_drawdown"]
    _SCALE = {"ann_ret_net": 100, "ann_vol_net": 100, "sharpe_net": 1,
              "sortino_net": 1, "max_drawdown": 100}
    _TOL = {"ann_ret_net": 0.006, "ann_vol_net": 0.006, "sharpe_net": 0.006,
            "sortino_net": 0.006, "max_drawdown": 0.06}

    _LA = {"DG-GMV": "DG-GMV", "DG-Asym": "DG-Asym", "DG-HRP": "DG-HRP",
           "DG-Shrink": "DG-Shrink", "Fixed Gerber": "gerber_c0.5",
           "Ledoit-Wolf": "ledoit_wolf", "ANS": "ans", "HRP": "hrp",
           "Equal weight": "equal_weight"}
    # Panel B runs the same family: DG-GMV is the scalar-threshold arm and
    # DG-Group is DG-Shrink with the threshold indexed by asset class
    _LB = dict(_LA, **{"DG-Group": "tgroup_shrink", "DG-GMV": "tglobal",
                       "DG-Shrink": "tglobal_shrink", "DG-Asym": "tasym",
                       "DG-HRP": "hrp_gerber"})

    mg = pd.concat([pd.read_csv(RES / "mixed_group_metrics.csv"),
                    pd.read_csv(RES / "mixed_arms_metrics.csv")])
    mg = mg.drop_duplicates(["universe", "method"])
    _SRC = {"A": (lambda N: panel[N], _LA, "sector-balanced"),
            "B": (lambda N: mg[mg.universe == f"mix{N}"].set_index("method"),
                  _LB, "mixed sector+ETF")}

    for _tag, (_get, _map, _what) in _SRC.items():
        _chunk = _tbl[:_iB] if _tag == "A" else _tbl[_iB:]
        _seen, _bad = [], []
        for _lab, _vals in _rows(_chunk):
            if _lab not in _map:
                _bad.append((_lab, "unmapped row label"))
                continue
            _seen.append(_lab)
            for _i, _N in enumerate(SIZES):
                _row = _get(_N).loc[_map[_lab]]
                for _j, _c in enumerate(_COLS):
                    _paper = _vals[_i * 5 + _j]
                    _data = _row[_c] * _SCALE[_c]
                    if abs(_paper - _data) > _TOL[_c]:
                        _bad.append((_lab, _N, _c, _paper, round(_data, 4)))
        claim(f"Table 1 Panel {_tag} ({_what}): all "
              f"{len(_seen) * 10} cells match the CSVs {_bad[:3]}",
              not _bad and len(_seen) == len(_map))

    print("\n== Table 1 Panel B: the mixed-universe claims ==")
    _gp = pd.read_csv(RES / "mixed_group_params.csv")
    _gd = pd.read_csv(RES / "mixed_group_dm.csv")
    _MIX = [f"mix{N}" for N in SIZES]
    _gs = _gp[(_gp.arm == "tgroup_shrink") & _gp.universe.isin(_MIX)
              & (_gp.year >= 2019)]
    check("group+blend: equity threshold, low", 1.7, _gs.c_eq.min(), tol=0.05)
    check("group+blend: equity threshold, high", 2.7, _gs.c_eq.max(), tol=0.05)
    check("group+blend: sleeve threshold, low", 0.18, _gs.c_slv.min(), tol=0.005)
    check("group+blend: sleeve threshold, high", 0.32, _gs.c_slv.max(), tol=0.005)
    claim("the two groups never overlap in any deployment year",
          (_gs.c_eq > _gs.c_slv).all())
    # the single threshold does not sit between the two regimes: it lands
    # inside the sleeve's own band, which is the point the text now makes
    _tg = _gp[(_gp.arm == "tglobal") & _gp.universe.isin(_MIX)
              & (_gp.year >= 2019)]
    check("single global threshold, low", 0.13, _tg.c_eq.min(), tol=0.005)
    check("single global threshold, high", 0.28, _tg.c_eq.max(), tol=0.005)
    claim("the single threshold lands inside the sleeve's own band, not "
          "above it",
          _gs.c_slv.min() <= _tg.c_eq.mean() <= _gs.c_slv.max()
          and "inside the region the sleeve itself prefers" in TEX)
    claim("...and far below the equity coefficient the joint fit selects",
          _tg.c_eq.max() < _gs.c_eq.min())
    # grouping alone barely lifts the equity coefficient
    _tgr = _gp[(_gp.arm == "tgroup") & _gp.universe.isin(_MIX)
               & (_gp.year >= 2019)]
    check("grouping alone: equity coefficient, low", 0.21,
          _tgr.c_eq.min(), tol=0.005)
    check("grouping alone: equity coefficient, high", 0.66,
          _tgr.c_eq.max(), tol=0.005)
    claim("grouping alone leaves the equity coefficient an order of "
          "magnitude below the joint fit", _tgr.c_eq.max() < _gs.c_eq.min())

    for _u in _MIX:
        _p_id = _gd[(_gd.universe == _u) & (_gd.method == "tgroup")
                    & (_gd.vs == "tglobal")].p.iloc[0]
        claim(f"{_u}: grouping alone is NOT identified (p={_p_id:.2f})",
              _p_id > 0.4 and f"{_p_id:.2f}" in TEX)
        _p_fx = _gd[(_gd.universe == _u) & (_gd.method == "tgroup_shrink")
                    & (_gd.vs == "gerber_c0.5")].p.iloc[0]
        claim(f"{_u}: group+blend beats the published rule at 1% "
              f"(p={_p_fx:.5f})", _p_fx < 0.01)
        _s = mg[mg.universe == _u].set_index("method")
        claim(f"{_u}: group+blend has the lowest volatility of any method",
              _s.drop(index=[i for i in _s.index if i.startswith("g2_")])
              .ann_vol_net.idxmin() == "tgroup_shrink")
        claim(f"{_u}: group+blend improves on the published rule's Sharpe",
              _s.loc["tgroup_shrink", "sharpe_net"]
              > _s.loc["gerber_c0.5", "sharpe_net"])
    # the 2x2 the paper now argues: neither ingredient works alone
    for _u, _p_sh, _p_gr in (("mix30", 0.044, 0.0058), ("mix70", 0.024, 0.0028)):
        _a = _gd[(_gd.universe == _u) & (_gd.method == "tgroup_shrink")
                 & (_gd.vs == "tglobal_shrink")].p.iloc[0]
        _b = _gd[(_gd.universe == _u) & (_gd.method == "tgroup_shrink")
                 & (_gd.vs == "tgroup")].p.iloc[0]
        check(f"{_u}: DG-Group over DG-Shrink, p", _p_sh, _a, tol=0.0006)
        check(f"{_u}: DG-Group over grouping alone, p", _p_gr, _b, tol=0.0006)
        claim(f"{_u}: DG-Group beats each one-ingredient arm at 5% or better",
              max(_a, _b) < 0.05)
    claim("both interaction tests are quoted in the text",
          "$p=0.044$ and $0.024$ against DG-Shrink" in TEX
          and "$p=0.0058$ and $0.0028$ against grouping alone" in TEX)
    claim("DG-Shrink alone lowers volatility but not Sharpe past the rule",
          all(mg[mg.universe == _u].set_index("method")
              .loc["tglobal_shrink", "ann_vol_net"]
              < mg[mg.universe == _u].set_index("method")
              .loc["gerber_c0.5", "ann_vol_net"]
              and mg[mg.universe == _u].set_index("method")
              .loc["tglobal_shrink", "sharpe_net"]
              < mg[mg.universe == _u].set_index("method")
              .loc["gerber_c0.5", "sharpe_net"] for _u in _MIX))

    # the paper claims the outright Sharpe lead only at n=70
    _cmp = ["tgroup_shrink", "tglobal", "gerber_c0.5", "ledoit_wolf",
            "ans", "hrp", "equal_weight"]
    _cmp = _cmp + ["tasym", "hrp_gerber"]
    claim("group+blend improves on the published rule's Sharpe and Sortino "
          "at both sizes",
          all(mg[mg.universe == _u].set_index("method")
              .loc["tgroup_shrink", _c]
              > mg[mg.universe == _u].set_index("method")
              .loc["gerber_c0.5", _c]
              for _u in _MIX for _c in ("sharpe_net", "sortino_net")))
    # the two leads the text concedes at n=70
    _m70 = mg[mg.universe == "mix70"].set_index("method").loc[_cmp]
    _m30 = mg[mg.universe == "mix30"].set_index("method").loc[_cmp]
    claim("at n=70 DG-Asym draws down less and DG-HRP earns more per unit "
          "of risk, which the text concedes",
          _m70.max_drawdown.idxmax() == "tasym"
          and _m70.sharpe_net.idxmax() == "hrp_gerber"
          and "DG-Asym draws down less and DG-HRP earns more per unit" in TEX)
    claim("at n=30 the group arm still leads drawdown and Sortino",
          _m30.max_drawdown.idxmax() == "tgroup_shrink"
          and _m30.sortino_net.idxmax() == "tgroup_shrink")
    # DG-HRP does not repeat its Panel A gain in the mixed design
    _ad = pd.read_csv(RES / "mixed_arms_dm.csv")
    for _u, _p in (("mix30", 0.30), ("mix70", 0.47)):
        _r = _ad[(_ad.universe == _u) & (_ad.method == "hrp_gerber")
                 & (_ad.vs == "hrp")].iloc[0]
        check(f"{_u}: DG-HRP vs plain HRP, p", _p, _r.p, tol=0.006)
        claim(f"{_u}: and it does not separate from plain HRP", _r.p > 0.10)
    claim("Panel A is where DG-HRP does beat plain HRP, at both sizes",
          all(panel[N].loc["DG-HRP", "sharpe_net"]
              > panel[N].loc["hrp", "sharpe_net"] for N in SIZES))
    for _u, _p in (("mix30", 0.038), ("mix70", 0.222)):
        _r = _ad[(_ad.universe == _u) & (_ad.method == "tasym")
                 & (_ad.vs == "gerber_c0.5")].iloc[0]
        check(f"{_u}: DG-Asym vs the published rule, p", _p, _r.p, tol=0.006)

    print("\n== Algorithm 1 against the code it describes ==")
    _alg = TEX[TEX.index(r"\begin{algorithm}"):TEX.index(r"\end{algorithm}")]
    for _piece in (r"\operatorname{sg}(m-x)", r"\operatorname{sg}(\nu-n)",
                   r"T - \widetilde n^{\top}\widetilde n",
                   r"(1-\delta)S_e", r"\lambda\bar\sigma_e^2 I"):
        claim(f"algorithm carries {_piece}", _piece in _alg)
    claim("the algorithm anneals the temperature across epochs",
          r"\tau \gets \tau_k" in _alg)
    claim("deployment is stated in terms of the hard states, not the tildes",
          "m^{\\top}m/(T-\\nu^{\\top}\\nu)" in _alg)
    # the blend conventions the algorithm's general form does not disambiguate
    claim("every arm in Table 1 is introduced in the comparator paragraph",
          all(_a in TEX.split("Table \\ref{tabmain}")[0]
              for _a in ("DG-GMV", "DG-Asym", "DG-Shrink", "DG-HRP",
                         "DG-Group")))
    claim("the blend's extra reach past the constant-threshold ceiling is "
          "stated",
          "DG-Shrink passes that bound at 14.26 and 13.64" in TEX)
    _sc = pd.read_csv(RES / "sector_ceiling_metrics.csv")
    claim("...and it is true: DG-Shrink is below the grid-best at both sizes",
          all(panel[N].loc["DG-Shrink", "ann_vol_net"]
              < _sc[_sc.N == N].ann_vol_net.min() for N in SIZES))
    claim("the text says which arms hold delta=1, learn it, or skip it",
          "DG-GMV and DG-Asym hold $\\delta=1$" in TEX
          and "from a neutral start at one half" in TEX
          and "bypasses the blend entirely" in TEX)
    _sh = pd.read_csv(RES / "headline_thresholds.csv")
    _d = _sh[(_sh.arm == "DG-Shrink") & _sh.N.isin(SIZES)].delta.dropna()
    claim(f"the learned blend does move off the neutral start "
          f"({_d.min():.2f}--{_d.max():.2f})", (_d != 0.5).all())

    print("\n== the two deployments the narrative now leads with ==")
    claim("DG-HRP beats 1/N on Sharpe AND Sortino at both sector sizes",
          all(panel[N].loc["DG-HRP", _c]
              > panel[N].loc["equal_weight", _c]
              for N in SIZES for _c in ("sharpe_net", "sortino_net")))
    claim("...and the quoted values match the panel",
          "0.74 and 1.04 at $n=30$ and 0.65 and 0.90 at $n=70$" in TEX)
    claim("the scalar arm is framed as the verifiable case, not the best one",
          "the case an exact finite-difference optimizer can also solve" in TEX
          and "The primary arm DG-GMV" not in TEX
          and "The primary model learns a positive scalar" not in TEX)
    claim("the abstract carries the risk-adjusted equity result",
          "takes the best Sharpe and Sortino of any method there, ahead of "
          "equal weighting" in TEX)
    claim("the conclusion names both deployments",
          "gain\nconcentrates in two deployments" in TEX
          or "gain concentrates in two deployments" in TEX)
    # the concession that DG-Shrink is a pure risk arm must stay honest
    claim("DG-Shrink does give up return and Sharpe to the rule at n=70",
          panel[70].loc["DG-Shrink", "ann_ret_net"]
          < panel[70].loc["gerber_c0.5", "ann_ret_net"]
          and panel[70].loc["DG-Shrink", "sharpe_net"]
          < panel[70].loc["gerber_c0.5", "sharpe_net"]
          and "gives up return and Sharpe in exchange" in TEX)
    claim("DG-Shrink does hold the shallowest drawdown at n=70",
          panel[70].max_drawdown.idxmax() == "DG-Shrink")

    print("\n== abstract and prose against the printed tables ==")
    _tabs = "".join(re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
                               TEX, flags=re.S))
    _tabnums = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", _tabs))
    _abs = TEX[TEX.index(r"\begin{abstract}"):TEX.index(r"\end{abstract}")]
    _absnums = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", _abs))
    claim(f"every number the abstract prints also appears in a table "
          f"{sorted(_absnums - _tabnums)}", not _absnums - _tabnums)
    # superlatives the prose asserts, re-derived rather than spot-checked
    _pa = {N: panel[N] for N in SIZES}
    _B = ["tglobal", "tasym", "hrp_gerber", "tglobal_shrink", "tgroup_shrink",
          "gerber_c0.5", "ledoit_wolf", "ans", "hrp", "equal_weight"]
    _pb = {u: mg[mg.universe == u].set_index("method").loc[_B] for u in _MIX}
    claim("prose: DG-HRP holds the best Sharpe and Sortino in Panel A",
          all(_pa[N].sharpe_net.idxmax() == "DG-HRP"
              and _pa[N].sortino_net.idxmax() == "DG-HRP" for N in SIZES))
    claim("prose: DG-GMV improves return, Sharpe and Sortino over the rule",
          all(_pa[N].loc["DG-GMV", _c] > _pa[N].loc["gerber_c0.5", _c]
              for N in SIZES
              for _c in ("ann_ret_net", "sharpe_net", "sortino_net")))
    claim("prose: 1/N takes the highest return and deepest drawdown in "
          "both panels",
          all(_pa[N].ann_ret_net.idxmax() == "equal_weight"
              and _pa[N].max_drawdown.idxmin() == "equal_weight" for N in SIZES)
          and all(_pb[u].ann_ret_net.idxmax() == "equal_weight"
                  and _pb[u].max_drawdown.idxmin() == "equal_weight"
                  for u in _MIX))
    _ratio = [_pb[u].loc["equal_weight", "ann_vol_net"]
              / _pb[u].loc["tgroup_shrink", "ann_vol_net"] for u in _MIX]
    claim(f"prose: 1/N runs nearly four times the learned volatility "
          f"({_ratio[0]:.1f}x, {_ratio[1]:.1f}x)",
          all(3.5 < _r < 4.2 for _r in _ratio))

    print("\n== sector constant-threshold ceiling ==")
    _sc = pd.read_csv(RES / "sector_ceiling_metrics.csv")
    _ss = pd.read_csv(RES / "sector_ceiling_summary.csv").set_index("N")
    for _N, _best, _learn in ((30, 14.30, 14.33), (70, 13.74, 13.76)):
        _g = _sc[_sc.N == _N].set_index("c")
        check(f"N={_N}: best constant threshold", 1.0, _g.ann_vol_net.idxmin(),
              tol=1e-9)
        check(f"N={_N}: ceiling volatility (%)", _best,
              _g.ann_vol_net.min() * 100, tol=0.006)
        check(f"N={_N}: learned volatility (%)", _learn,
              panel[_N].loc["DG-GMV", "ann_vol_net"] * 100)
        _share = ((panel[_N].loc["gerber_c0.5", "ann_vol_net"]
                   - panel[_N].loc["DG-GMV", "ann_vol_net"])
                  / (panel[_N].loc["gerber_c0.5", "ann_vol_net"]
                     - _g.ann_vol_net.min()))
        claim(f"N={_N}: the learner captures about four fifths of the "
              f"available margin ({_share:.0%})", 0.72 < _share < 0.88)
        claim(f"N={_N}: the objective is not flat -- c=2.5 exceeds 21%",
              _g.loc[2.5, "ann_vol_net"] * 100 > 21)
    claim("the hindsight optimum is 1.0 in the sector universes and in "
          "top-100 alike",
          set(_ss.best_constant_c) == {1.0}
          and abs(g.loc["c=1.0", "ann_vol_net"] * 100 - 13.72) <= 0.006)
    claim("the text reports the sector ceiling",
          "hindsight-best fixed threshold is 1.0 at both sizes" in TEX
          and "four fifths of the available margin" in TEX)

    print("\n== universe audit: the manuscript reports N in {30, 70} only ==")
    # every equity-universe size named in the body, in any notation
    _sizes = {int(s) for s in re.findall(r"\$n=([0-9]+)\$", TEX)}
    _sizes |= {int(s) for pair in re.findall(r"n\\in\\\{([0-9,]+)\\\}", TEX)
               for s in pair.split(",")}
    claim(f"no equity-universe size outside {{30, 70}} is named {sorted(_sizes)}",
          _sizes == {30, 70})
    claim("the sector design is declared primary and carries both sizes",
          r"The primary design is sector-balanced" in TEX
          and r"n\in\{30,70\}" in TEX)
    claim("the mixed design is tied to the same equity selection",
          "adds a fixed nine-ETF sleeve to the same equity selection" in TEX)
    claim("Table 1 declares that Panel B headings are the equity count",
          "column headings give the equity count" in TEX)
    claim("top-100 is presented as a second cross-section, not the primary",
          "supplies a second cross-section" in TEX
          and "primary universe" not in TEX)
    for _tag, _panel in (("Panel A", "Sector-balanced equities"),
                         ("Panel B", "nine-ETF multi-asset sleeve")):
        claim(f"Table 1 {_tag} is labelled {_panel!r}", _panel in TEX)
    # both designs must be argued in the prose, not just tabulated
    claim("the sector panel is discussed at both sizes",
          "14.43 to 14.33 percent at $n=30$" in TEX
          and "13.86 to 13.76 at $n=70$" in TEX)
    claim("the mixed panel is discussed at both sizes",
          "4.31 to 3.92 and 4.46 to 4.00 percent" in TEX
          and "0.73 against 0.69 and 0.68 against 0.67" in TEX)
    # both panels must name their arms from the same family
    claim("Panel B reuses the DG family rather than inventing labels",
          "DG-Group" in TEX and "Learned $c$ (single)" not in TEX)
    claim("both panels now carry the same arms",
          set(_LA) - {"DG-GMV", "DG-Asym", "DG-HRP", "DG-Shrink"}
          == set(_LB) - {"DG-GMV", "DG-Asym", "DG-HRP", "DG-Shrink",
                         "DG-Group"})
    claim("the vector threshold DG-Group needs is defined in the method",
          "construction admits a vector threshold" in TEX)
    claim("the caption relates DG-Group to DG-Shrink",
          "DG-Group extends DG-Shrink with one threshold per asset class"
          in TEX)

    print("\n== top-100 numbers the prose still prints ==")
    _t100full = pd.concat([t100, pd.read_csv(RES / "ew_baseline_metrics.csv")
                           .query("universe=='top100' and cost_bps==10")
                           .set_index("method").loc[["equal_weight"]]])
    for _key, _lbl, _v in (("A1_gmv_tglobal", "DG-GMV", 13.76),
                           ("gerber_c0.5", "published rule", 14.04),
                           ("ledoit_wolf", "linear shrinkage", 14.62),
                           ("ans", "ANS", 14.27),
                           ("equal_weight", "1/N", 17.78)):
        check(f"top-100 volatility (%), {_lbl}", _v,
              _t100full.loc[_key, "ann_vol_net"] * 100)

    _pC = {  # consistency table: c-bar, vol
        "A_ste_hard": (1.09, 13.76), "E_num_only": (1.40, 14.01),
        "B_soft_soft": (1.46, 13.95), "C_soft_deployhard": (1.46, 14.01),
        "D_finite_diff": (0.99, 13.72),
    }
    _okC = all(abs(fbi.loc[_k, "c_mean_2019plus"] - _c) <= 0.005
               and abs(fbi.loc[_k, "ann_vol_net"] * 100 - _v) <= 0.006
               for _k, (_c, _v) in _pC.items())
    claim("Table 2 Panel A: the five optimization routes match the CSVs", _okC)
    claim("Table 2 Panel A: grid 13.92 and hindsight 13.72 match",
          abs(g.loc["val_select", "ann_vol_net"] * 100 - 13.92) <= 0.006
          and abs(g.loc["c=1.0", "ann_vol_net"] * 100 - 13.72) <= 0.006)

    # the caption pairs stars with named comparators; check those tests
    _p_lw = d100[(d100.method == "diffgerber_tglobal")
                 & (d100.vs == "ledoit_wolf")].p.iloc[0]
    claim(f"top-100 prose: DG-GMV vs Ledoit-Wolf at 1% (p={_p_lw:.5f})",
          _p_lw < 0.01)
    # one comparator throughout: every minimum-variance arm against the
    # published fixed rule.  Panel A earns no star, and the text says so.
    _pa = dm[(dm.vs == "gerber_c0.5")
             & dm.method.isin(["DG-GMV", "DG-Asym", "DG-Shrink"])]
    claim(f"no Panel A arm separates from the published rule "
          f"(p from {_pa.p.min():.2f} to {_pa.p.max():.2f})",
          _pa.p.min() > 0.10 and len(_pa) == 6)
    check("Panel A, smallest p against the published rule", 0.375,
          _pa.p.min(), tol=0.006)
    check("Panel A, largest p against the published rule", 0.970,
          _pa.p.max(), tol=0.006)
    claim("Panel A carries no volatility stars",
          "$13.76^{**}$" not in TEX and "\\mathbf{13.64}^{***}" not in TEX)
    claim("the text explains the unstarred panel",
          "which is why the panel is unstarred" in TEX)
    claim("the caption states one meaning for the star throughout",
          "carry one meaning throughout" in TEX)
    claim("the caption warns the panels differ in risk level",
          "not comparable in level" in TEX)
    # the rival-shrinkage comparisons move to prose and must stay true
    _p_ans = dm[(dm.method == "DG-Shrink") & (dm.vs == "ans")
                & (dm.N == 70)].p.iloc[0]
    claim(f"DG-Shrink vs ANS at 1% for N=70, as the text states "
          f"(p={_p_ans:.5f})", _p_ans < 0.01)

    # the state-network star is a comparison WITH THE SCALAR LEARNER
    _fl = pd.read_csv(RES / "fold_losses_gs2022.csv").pivot(
        index="date", columns="method", values="loss")
    from diffgerber import stats as _st
    _dm_ts, _p_ts = _st.diebold_mariano(
        _fl["diffgerber_tstate"].values, _fl["diffgerber_tglobal"].values)
    claim(f"state network vs scalar learner reaches the 10% level "
          f"(DM={_dm_ts:+.2f}, p={_p_ts:.4f})", 0 < _p_ts < 0.10 and _dm_ts > 0)

print(f"\n{'=' * 70}\nPASS {ok}   FAIL {bad}")
raise SystemExit(1 if bad else 0)
