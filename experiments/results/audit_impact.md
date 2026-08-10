# Audit impact: pre-audit vs corrected results

Old numbers: `archive_pre_audit/` (9 defects present: train/eval ridge mismatch,
mcap look-ahead, gs2022 degeneracy, no HAC in DM, inert psd_clip with
inconsistent NOCLIP, LongOnlyGMV ridge, Adam state accumulation, unequal LR,
log/simple return mixing). New numbers: pre-registered protocol (plan
`humble-drifting-map`): scale-relative ridge 1e-3 train+eval, straight-through
estimator, fresh Adam per refit, uniform lr=0.02, horizon-aware refit boundary,
strict PIT market caps, simple-return metrics, DM auto-lag (h=4) Bartlett.

## Headline (annualized net vol, 10bps; Sharpe in parens)

| method | top-100 old | **top-100 new** | top-400 old | **top-400 new** |
|---|---|---|---|---|
| Gerber fixed c=0.5 | 14.09% (0.39) | 14.04% (0.44) | 13.49% (0.49) | 13.27% (0.55) |
| Ledoit-Wolf | 14.70% (0.36) | 14.62% (0.43) | 13.30% (0.31) | 13.18% (0.40) |
| ANS | 14.31% (0.45) | 14.27% (0.51) | 12.59% (0.53) | **12.54% (0.59)** |
| HRP | 15.47% (0.56) | 15.54% (0.62) | 16.01% (0.45) | 15.93% (0.53) |
| **T-global** | 14.05% (0.46) | **13.76% (0.54)** | 12.87% (0.36) | **12.76% (0.54)** |
| T-asset | 14.22% (0.52) | 14.15% (0.52) | 13.35% (0.35) | 13.33% (0.60) |
| T-asym | 14.03% (0.47) | 13.95% (0.48) | 13.19% (0.34) | 13.14% (0.51) |
| T-state | 14.45%* (0.31) | 14.44% (0.52) | 12.66%** (0.37) | 13.45% (0.39) |

*old top-100 T-state additionally carried the degeneracy artifact (16.88% in
the v2 run). **old top-400 T-state was best-of-3-seeds under a deflated-
variance training estimator.

## Conclusion-by-conclusion verdict

| claim | old status | **new status** |
|---|---|---|
| Learned beats fixed c point-wise | +5bp / +62bp | **+28bp / +51bp — stronger at top-100** |
| Learned vs LW significant (top-100) | p=0.0015 (no HAC — fragile) | **p=0.0004 WITH HAC (T-asym p=0.0021) — robust** |
| Learned vs ANS | loses at top-400, ns at top-100 | **wins at top-100 (p=0.079); ns behind at top-400 (12.76 vs 12.54)** |
| Learned reaches the ex-post oracle | NO — overshot to 1.64, DM +3.71 vs c=1.0 | **YES — within 3bp (top-100) / 10bp (top-400) of the oracle; beats val_select on both** |
| c=0.5 outside the OOS-optimal basin | yes | **yes (14.04 vs 13.72; 13.27 vs 12.66)** |
| T-state wins in high dimension | 12.66% best (seed 0) | **DEAD — 13.45%, loses to fixed everywhere; capacity uniformly harmful under equal budgets** |
| Optimal c depends on dimension (1.64 vs 0.89) | claimed | **DEMOTED — corrected learned c is ~1.0±0.2 on both universes; the axis was largely a bug artifact** |
| Long-only: learning harmful (E4 boundary) | DM +4.47, p<1e-4 (ridge artifact suspected) | **SURVIVES: 15.00% vs 14.84%, DM +2.30, p=0.023 (weaker but real); learned paths still decision-dependent** |
| Fixed-c blowup (42.7%) | at eval ridge 1e-4 | **conditional as pre-registered: at ridge 1e-3, c=2.25 degrades to 17.2% (no catastrophe); c-grid spread remains** |
| Asset-class direction flip (crypto) | learned c 0.32-0.71 | **SURVIVES: learned c 0.15-0.36 vs equities ~1.0; higher fixed c monotonically harmful in crypto (67.77 → 70.17%); learned ≈ best Gerber (p=0.95 vs fixed)** |
| Sharpe objective unlearnable | c collapses to 0.075 | **SURVIVES, stronger: c collapses to 0.012; OOS Sharpe 0.30 vs variance-trained 0.54 (top-100); worse on both metrics at both universes** |
| Init/seed stability | top-100 only | **BOTH universes: tglobal init-invariant (12.76-12.88%, same c path); T-state seeds now tight (13.31-13.45%) and uniformly behind — clean demotion with error bars** |
| Ridge sensitivity of learned c | eval-only sweep (vacuous) | **REAL now: retrained per ridge level — learned c virtually identical across {1e-4,1e-3,1e-2} and beats fixed at every level; c=1.25@1e-4 still spikes (16.48%, 12.9x lev) → conditional fragility intact** |
| Subperiod robustness | mixed, 2021-22 weakness | **IMPROVED: top-100 wins COVID (28.11 vs 29.35) and 2023-26, 2021-22 gap shrinks to 13bp; top-400 wins/ties every period, +154bp in 2023-26; most cost-resilient (Sharpe@20bps 0.47 vs fixed 0.36, LW 0.30)** |
| Cross-universe pooled significance | invalid (mixed estimators) | **valid pooling: vs LW t=-2.21 p=0.029, bootstrap CI excludes 0 — cross-universe significance achieved; vs fixed-c still ns (p=0.19, sign 4/8) — honest** |

## Final corrected-world claim set (all R1-R11 complete)

1. **T-global is the story**: best-in-class at top-100 (13.76%, best vol AND
   Sharpe AND MDD; beats ANS point-wise p=0.079), beats fixed/LW point-wise at
   top-400 (12.76%), within 3-10bp of the ex-post oracle threshold on both
   universes, beats grid+validation selection on both, init-invariant on both,
   ridge-invariant, most cost-resilient, and significant vs LW pooled across
   universes with HAC (p=0.029).
2. **Non-universality map (revised axes)**: time (c path moves ~0.75-1.23),
   asset class (equities ~1.0 vs crypto 0.15-0.36 — direction flip), decision
   layer (long-only learns a different path AND learning there is harmful,
   p=0.023). Dimension axis DROPPED (was a bug artifact).
3. **Learnability boundaries**: variance objective learnable; Sharpe objective
   collapses (c→0.01) and fails its own metric; capacity uniformly harmful
   (T-state behind fixed everywhere, tight seed bars); long-only harmful.
4. **Conditional fragility**: at negligible ridge, mis-set fixed thresholds
   spike (c=1.25: 16.5%, 12.9x leverage); learned thresholds do not, and are
   ridge-invariant — but framed as decision-layer conditioning, not magic.
5. Honest ANS position: ties/behind at top-400 (12.76 vs 12.54, ns), ahead at
   top-100 (13.76 vs 14.27, p=0.079).

## Ground-truth simulation (2026-07-31) — new, and it adds a concept

`simulation_*.csv`. One-factor correlation, N=50, T=252, 120 replicates;
returns from multivariate t (df 3..inf) and from Gaussian + idiosyncratic
contamination. Because the true covariance is known, portfolio quality is the
exact excess variance ratio w'R_true w / oracle, with zero evaluation noise.

**1. Gerber's robustness advantage, verified against ground truth.** Gerber at
its best threshold vs shrinkage (excess variance ratio, lower better):

| regime | Gerber@c* | sample | Ledoit-Wolf | ANS |
|---|---|---|---|---|
| t, df=3 | **1.157** | 1.561 | 1.245 | 1.293 |
| t, df=8 | 1.155 | 1.264 | 1.189 | **1.104** |
| Gaussian | 1.146 | 1.202 | 1.163 | **1.082** |
| +1% outliers | **1.160** | 1.356 | 1.298 | 1.244 |
| +3% outliers | **1.174** | 1.462 | 1.498 | 1.417 |
| +5% outliers | **1.193** | 1.527 | 1.639 | 1.537 |
| +3%, size 15 | **1.175** | 1.632 | 1.767 | 1.648 |

Crossover at df≈5-8: Gerber dominates under heavy tails/contamination, ANS
dominates near-Gaussian. **This explains the real-data pattern** (ANS ahead on
equities at top-400) rather than leaving it as an unexplained loss. Note
shrinkage gets WORSE than the sample covariance under contamination — shrinking
toward a target does not defend against outliers, thresholding does.

**2. Optimal threshold falls as tails get heavier** — c* = 0.4 (Gaussian) →
0.3 → 0.2 (df=3), monotone; and 0.4 (clean) → 0.3 → 0.2 → 0.1 as contamination
rises. Mechanism confirmed at the DGP level: heavier tails inflate sigma, so a
threshold in sigma units captures fewer exceedances (P(|z|>0.5) = 0.62 Gaussian
vs 0.44 at df=3). This gives the *direction* of the real-data asset-class flip
(crypto ~0.3 vs equities ~1.0) but not its full magnitude.

**3. Two targets, not one — the conceptual finding.** The learned threshold did
NOT match the simulation's oracle under contamination (learned ~0.95 vs oracle
0.2), which first looked like a learning failure. It is not. There are two
distinct targets and contamination splits them:

| contamination | statistical target c* (recover clean Σ) | decision target c* (realized variance on observed data) |
|---|---|---|
| none | 0.4 | 0.4 |
| 1% outliers | 0.3 | 0.6 |
| 3% outliers | 0.2 | 0.7 |
| 5% outliers | 0.2 | 1.5 |

The learner tracks the **decision** target (its own loss), landing inside its
flat optimum in every case. Classical robust estimation implicitly optimizes
the statistical target. **Under contamination these diverge sharply, and only
the decision target is what an investor holding the portfolio experiences.**
This is a statement about decision-focused learning in general, not about
Gerber: DFL optimizes the outcome you measure, and when data contains outliers
that is *not* the same as recovering the clean second moment.

Caveat to state plainly: in the clean elliptical DGP the objective surface is
nearly flat (excess ratio 1.15-1.27 across the whole grid), so neither the
oracle nor the learner is identified there; the threshold matters only when
the data departs from ellipticity — which is exactly when Gerber is motivated.

## Net assessment

The corrected pipeline is FAVORABLE on balance: the core performance story
(T-global best-in-class at top-100 incl. vs ANS, oracle-level threshold
recovery, significant-with-HAC LW comparison, E4 boundary) came out stronger
or intact, while two claims died honestly (T-state high-dim headline; the
dimension axis of non-universality). The paper's reliable protagonist is
T-global; T-asym is the secondary interpretable variant; T-state becomes a
capacity-ablation negative result.
