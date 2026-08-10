# Phase 4 interim results — 2026-07-29

Protocol identical to pilot: point-in-time top-100, window 252, step 21,
eval 2017-2026 (115 folds), annual expanding refit, 10bps costs.

## Threshold-module grid (GS2022 family)

| method              | ann vol | Sharpe | MDD    | turnover |
|---------------------|---------|--------|--------|----------|
| Gerber c=0.5 (fixed)| 14.10%  | +0.38  | -31.7% | 0.88     |
| Ledoit-Wolf         | 14.70%  | +0.36  | -31.1% | 1.55     |
| T-global (learned)  | 14.05%  | +0.46  | -31.2% | 0.76     |
| T-asset (learned)   | 14.22%  | +0.52  | -32.3% | 0.72     |
| **T-asym (learned)**| **14.03%** | +0.47 | -31.3% | 0.80   |
| T-state (learned)   | 14.45%  | +0.30  | -30.7% | 0.89     |

DM vs LW: T-global p=0.0015, T-asym p=0.0052 (significant). None significant
vs fixed c=0.5. **More capacity did not help OOS**: scalar-level learning
(T-global/T-asym) > per-asset > state-MLP. cos family uniformly weaker
(best 15.13%); within-family learning helps (T-asset vs fixed cos p=0.0003).

**T-asym found an interpretable asymmetry**: post-2021 it settles at
c_up ≈ 0.95, c_down ≈ 1.69 — moderate upside co-moves count, only extreme
downside co-moves count. Novel empirical finding (no asymmetric Gerber in
the literature); connects to asymmetric-dependence literature.

## Threshold-sensitivity grid + validation-select (GS2022)

OOS ann vol by fixed c: 0.25→14.32, 0.5→14.09, 0.75→13.96, **1.0→13.77**,
1.25→13.88, 1.5→14.02, 1.75→13.95, 2.0→14.15, 2.25→14.26.
Flat basin ≈ [0.75, 1.75]; published default c=0.5 is outside the basin.

- val_select (per-year argmin over grid on trailing folds): 13.95%, +0.52.
- gradient-learned T-global: 14.04%, +0.47. learned vs val_select: p=0.17 (ns).
- Both adaptive methods land in the basin without oracle access, but neither
  recovers the single ex-post-best point (c=1.0): learned vs c=1.0 DM +3.71
  (p=0.0003). The learned c overshot to ~1.64 post-COVID (training window
  dominated by 2020-2021; ex-post eval optimum drifted to ~1.0).

## Honest read for the paper

1. **The threshold critique lands hard**: c=0.5 is outside the OOS-optimal
   basin; every c in [0.75, 1.75] beats it. Threshold choice matters (55bps
   vol spread) and the optimum is unknowable a priori.
2. **Decision-learning finds the basin without an oracle** and significantly
   beats LW/sample/EWMA — but within the flat basin, gradient learning ≈
   grid validation ≈ any fixed basin point (differences ns). The
   *differentiability* contribution must therefore be sold as (a) the
   framework enabling structured variants impossible for grid search
   (asymmetric, per-asset, state, and end-to-end integrations — grid search
   over 2+ coupled continuous parameters is exponential), and (b) the PSD
   theory; not as "gradient beats validation on a scalar".
3. **T-asym is the novelty sweet spot**: same performance as the best method,
   genuinely new estimator, interpretable asymmetry, and infeasible for
   naive 2-D grid+validation at fine resolution... (though a coarse 2-D grid
   is feasible — must add asym-grid control to claim this honestly).
4. Open: turnover-penalized objective, multi-seed stability, subperiods,
   asym 2-D grid control, ridge-sensitivity ablation.

## High-dimensional results (top-400, N=400 > T=252) — 2026-07-30

Training used DG_NOCLIP=1 (no PSD clip during training; eval is hard GS2022,
provably PSD). Learned T-global c ≈ 0.89 (vs 1.64 at top-100 — optimal
threshold depends on universe size too).

| method              | ann vol | Sharpe | turnover |
|---------------------|---------|--------|----------|
| Gerber c=0.5 (fixed)| 13.49%  | +0.49  | 2.18     |
| Ledoit-Wolf         | 13.30%  | +0.31  | 3.13     |
| HRP                 | 16.01%  | +0.45  | 0.34     |
| T-global            | 12.88%  | +0.36  | 1.93     |
| T-asset             | 13.35%  | +0.35  | 1.92     |
| T-asym              | 13.19%  | +0.34  | 1.97     |
| **T-state**         | **12.66%** | +0.37 | 1.79   |

- **All four learned variants beat fixed c=0.5 and LW point-wise**; T-state
  (useless at N=100) is best in high dim and edges even the ex-post-best
  fixed threshold (c=1.0: 12.70%). Individual DM not significant (p≈0.27),
  direction uniform 4/4.
- Threshold grid at top-400: basin is narrower and FRAGILE. c=1.25 →
  16.56% and c=2.25 → 42.71% (!) — diagnosed: on COVID-era windows the
  hard GS2022 matrix goes rank-deficient (min eig ≈ 1e-19), GMV weights
  explode to 13x gross leverage. Learned/adaptive thresholds never entered
  the degenerate region. **In high dimension a mis-set fixed threshold is
  not a bps cost — it is a blow-up risk.** (Caveat for paper: add
  ridge-sensitivity ablation; fragility statement is "at standard ridge".)
- Gradient-learned (12.87%) beats grid+validation val_select (13.09%, picked
  c=0.75) point-wise at top-400 — continuous optimization finds between-grid
  optima (c≈0.9); ns individually (p=0.53).

## Ablations ①② — 2026-07-30

**① Ridge sensitivity (top-400)**: larger ridge mitigates the fixed-c
catastrophe (c=2.25: 42.7% → 17.2% @1e-3 → 14.3% @1e-2) but badly-set fixed
thresholds remain 145bps+ worse than learned; the **learned threshold is
ridge-invariant** (12.87/12.86/12.85% across three ridge levels). Claim
survives with honest wording. (`ablation_ridge_gs2022_top400.csv`)

**② Multi-init / multi-seed stability (top-100)**: T-global from init
{0.25, 0.5, 1.0} converges to the same trajectory (c2020≈0.93, c2023≈1.65,
c2026≈1.64) and OOS vol 14.04-14.07% — initialization-invariant. T-state
across seeds {0,1,2}: vol 14.44-14.53%, similar c trajectories — no seed
luck. The pilot's 2026 dip to 1.18 did not reproduce under the main
protocol (was pilot-script warm-start path). (`ablation_stability_gs2022.csv`)

## Ablations (a)+③ — 2026-07-30

**(a) T-state v2 (7 features incl. term spread/drawdown/VIX momentum/
exceedance rate, hidden 32): NEGATIVE result.** top-100: 16.88% (v1 14.45%)
— clear overfitting; top-400: 12.83% (v1 12.66%) — slightly worse. The
small 3-feature/hidden-16 net remains the best state module. Paper framing:
"a compact state network suffices; added capacity overfits" (reported as
capacity ablation). v1 T-state at top-400 (12.66%) stands as headline.

**③ Combined cross-universe significance: pooling does NOT elevate to
significance vs fixed-c.** Pooled t: tglobal p=0.34, tasym p=0.34; sign
test 6/8 (p=0.14). vs LW: significant at top-100 only (tglobal p=0.0015,
tasym p=0.0052). Note ③ ran on v2 fold losses (tstate rows penalized by
the overfit v2; tglobal/tasym unaffected — identical to v1). Final paper
claims must be: direction-consistent point improvements + significant vs
LW at top-100 + robustness (ridge/init/seed) + fragility avoidance +
threshold non-universality. NOT "significantly beats fixed-c on variance".

## E-package (additional experiments E1-E5) — 2026-07-30

**E3 subperiod + costs** (`subperiod_analysis.csv`, `cost_sensitivity.csv`):
gains not single-period (top-100: COVID; top-400: 2023-26 + calm). Bonus:
T-asym wins 3/4 subperiods at top-100 incl. 2021-22 where T-global loses.
Costs: at top-100 learned degrades slower with costs (Sharpe@20bps 0.40-0.41
vs fixed 0.32, LW 0.24); at top-400 fixed-c has better Sharpe (vol/Sharpe
tension — report).

**E1 ANS** (`e1_ans.csv`): top-100 ANS 14.31% (we win); top-400 **ANS 12.59%
beats T-global 12.87% and ties T-state v1 12.66%** (DM ns). Headline must be
"matches SOTA with robust/interpretable/decision-adaptive estimator +
avoids fixed-c fragility", NOT "beats SOTA".

**E5 transfer** (`transfer_matrix.csv`): thresholds TRANSFER fine between
equity universes (in-situ ≈ transferred, p≈0.94) — non-transferability
hypothesis rejected within asset class. What is robust and portable is the
learned TIME PATH; fixed constants are the fragile objects (c=1.25 blowup).

**E2 crypto** (`crypto_results.csv`): learned c drops to 0.32-0.71 —
OPPOSITE direction vs equities (0.9-1.64). Fixed high c monotonically hurts
in crypto (c=1.5: 70.46% vs learned 67.38%, i.e. equity-level thresholds
cost ~310bps). Combined with E5: **thresholds transfer within an asset
class but flip direction across asset classes.** Honest note: in crypto no
Gerber variant beats ANS/sample (~67% all); the claim is about threshold
behavior, not SOTA there.

**E4 decision dependence** (`decision_dependence.csv`): the learned path
DOES depend on the decision layer (long-only learns a slower, lower path:
0.68→1.65 gradual vs long-short 0.9→1.73 with sharp COVID jump). BUT
under long-only, learning HURTS OOS: fixed 0.5 = 14.91% vs learned 15.19%
(DM +4.47, p<0.0001 — strongest stat in the study, against learning).
Long-only GMV is robust to covariance error, so threshold learning adds
noise. **Boundary condition: decision-focused threshold learning pays only
where the decision is covariance-sensitive.**

## Final round A/B/C — 2026-07-30

**A. G-hybrid (diagonal-normalized GS2022): DEAD as an estimator.** Despite
PSD-by-construction, it significantly underperforms GS2022+clip everywhere
(top-100 learned: 15.33% vs 14.04%, DM p=0.019; top-400: 15.6-20% vs
12.9-13.5%). Insight salvaged: **the pairwise denominator (T - nn_ij) is
load-bearing** — replacing it with its diagonal geometric-mean approximation
destroys performance. The theory-practice tension stands; frame G-cos/
G-hybrid as theoretical devices that localize WHERE GS2022's empirical
power comes from. (`ghybrid_results.csv`)

**B. T-state top-400 multi-seed: headline DEMOTED.** Seeds {0,1,2} give
12.64 / 13.12 / 13.22% — mean ≈ 12.99%, spread 58bps; the original 12.66%
was the best seed. Seed-mean still beats fixed (13.49) and LW (13.30) but
loses to ANS (12.59) and to T-global (12.87, which IS init-invariant).
**Reliable learned-variant claim = T-global; T-state reported with seed
error bars as promising-but-variable.** Seed-0 fold losses + snapshots
persisted as canonical v1 record. (`tstate_seeds_top400.csv`)

**C. Vol-targeting: fails as constructed — for everyone.** All estimators
massively overshoot the 10% target (realized 17-32%) because GMV's
predicted vol w'Σ̂w is biased low (optimization bias), worst for
high-threshold Gerber (top-100 tglobal realized 23.4% vs fixed 18.9%; ANS
best 17.2%). Lesson: our estimators improve GMV *structure* (relative
weights → realized OOS vol) but not absolute *scale* calibration; naive
model-predicted-vol targeting is broken for all methods. Do NOT claim
risk-targeting benefits; keep realized-OOS-vol evaluation and note the
scale-calibration issue honestly (or omit C). (`vol_targeting.csv`)

## ⚠️ Train/eval ridge inconsistency — found 2026-07-30, CORRECTS an earlier claim

**The bug**: training used an ABSOLUTE ridge (`Sigma + 1e-4 I`, = 76.7% of the
mean diagonal for daily returns) while evaluation uses a SCALE-RELATIVE ridge
(`Sigma + 1e-4 * tr/N * I`, = 0.01%). Training was ~7,700x more regularized
than deployment. Between-method comparisons remain fair (all methods share the
same evaluation path), but the learned thresholds were fit to the wrong
decision layer.

**Retrained with matched ridge** (`ridge_consistency.csv`):
- top-100: 14.04% → **13.94%** (DM -1.84, p=0.068). Fixing it HELPS; gap over
  fixed c=0.5 widens 4bp → 15bp. Learned c path converges to the same place.
- top-400: 12.87% → **18.38%**. Fixing it HURTS badly.

**Diagnosis of the top-400 reversal**: the damage is ONE fold. Year-by-year
realized vol is nearly identical between the two runs except 2020 (29.7% vs
19.3%), and the worst fold is 2020-04-07 at **138% annualized with 12.3x gross
leverage** (matched) vs 4.7x max (original). Cause: the matched-ridge training
learned c≈1.14 for 2019/2020 — trained on 2016-2019 folds that contain no
degenerate regime — and c≈1.14 sits in the fragility zone the grid already
identified (c=1.25 → 16.56%).

**Claim that must be RETRACTED**: "learned thresholds never enter the
degenerate region." That held only because the accidental heavy training ridge
kept learned c low. With a correctly specified objective, threshold learning
walks into the blowup when the training window lacks the degenerate regime.

**Claim that is STRENGTHENED**: the high-dimensional fragility is a property of
Gerber+GMV itself, not of careless human threshold choice — it catches learned
thresholds too. Decision-focused learning inherits the conditioning pathology
of its decision layer.

**Principled resolution — DONE** (`conditioning_control_top400.csv`). Same
scale-relative ridge in training AND evaluation for every method, swept:

| ridge | learned | fixed 0.5 | fixed 1.25 | LW | ANS |
|-------|---------|-----------|-----------|-----|-----|
| 1e-4 (orig eval) | 18.38%* | 13.48% | 16.56% | 13.30% | 12.59% |
| 1e-3 | 13.21% | 13.47% | 13.18% | 13.27% | 12.59% |
| 1e-2 | **12.99%** | 13.41% | 13.00% | 13.14% | **12.56%** |

*matched-ridge training at the negligible 1e-4 level; max gross leverage
12.3x → 5.8x (1e-3) → 4.9x (1e-2).

Consequences, all of which must be reflected in the paper:

1. **The blowup is a decision-layer conditioning artifact, not a threshold
   artifact.** With any sane ridge (≥1e-3 of the mean diagonal) the entire
   c-grid collapses into a ~13% band: the c=1.25 catastrophe (16.56%) becomes
   13.18%, and c=2.25's 42.71% disappears. Fragility claim must be restated as
   "an unregularized high-dimensional GMV amplifies threshold misspecification
   into blowups; adequate conditioning control removes this" — weaker as a
   scare story, stronger and more useful as guidance.
2. **Threshold learning's advantage shrinks to ~40bp and stays insignificant**
   (learned 12.99% vs fixed 13.41%, DM -0.70 p=0.48). Under proper
   conditioning, threshold choice matters much less than we claimed.
3. **ANS remains the best estimator at every ridge level** (12.56-12.59%) and
   is unaffected by conditioning — a robust, well-conditioned baseline. Our
   honest position at top-400 is "comparable to, not better than, ANS."
4. Learned c is stable across ridge levels (0.93-0.96 late years) — the
   learning itself is well-behaved; it was the decision layer that was not.

**Net effect on the paper**: the "fixed thresholds blow up" headline is
demoted to a conditional statement; the surviving core is the non-universality
map (time / asset class / dimension / decision), the learnability boundary
(long-only harmful, Sharpe unlearnable), the method-and-theory novelty, and a
new methodological lesson — decision-focused learning inherits its decision
layer's conditioning pathology, so the regularization must be specified
consistently and disclosed.

## ⚠️ Novelty claim correction — prior art on threshold tuning (verified 2026-07-30)

Our first literature survey answered "No" to "has anyone learned/adapted the
Gerber threshold". **That was wrong.** A targeted re-audit of the primary
sources found four works that select c from data:

| Work | Method | Objective | OOS outcome |
|---|---|---|---|
| **Zhou (2024)** arXiv:2406.00610 | grid c∈[0.3,1.0] + 5-fold CV | Sharpe | c*=0.6 (sd), 0.4 (MAD); modest gain |
| **Rubsamen (2023)** portfoliooptimizer.io | rolling walk-forward argmax | Sharpe (24m) | **no gain**: SR 1.26 → 1.23 |
| **Abu Khalaf & Smyth (2025)** arXiv:2512.23021 | Optuna TPE (GS benchmark) | after-cost Sharpe | **no gain**: θ*=0.439, SR 0.51 → 0.51 |
| **Smyth (2026)** Kernel-IQ, SSRN 6757938 | Optuna, 100 trials | MaxSR | unknown |

Note Zhou (2024) was already in our survey list but read only as a
benchmarking paper; the cross-validation of c was missed. It is discoverable
on the first page of a Google search — a reviewer would find it.

**Confirmed as we believed**: the two "dynamic Gerber" papers keep c fixed.
Algieri et al. (2021) make the nine co-exceedance *probabilities* time-varying
via CARML autoregressive logit (ML-estimated params are γ, α, β only;
thresholds enter as fixed inputs, c=0.5 or a fixed 90% quantile). Leccadito
et al. (2024) run a DCC-style recursion on count matrices (6 estimated params
a_C,b_C,a_D,b_D,a_N,b_N); thresholds are "half the unconditional volatility"
applied to GARCH-standardized innovations, i.e. the threshold *level* moves
with conditional vol but the *coefficient* is fixed. arXiv:2512.23021 fixes
its own gate at c=0.5 ("fixed at 0.5 in this study") while tuning the Gerber
benchmark's θ.

**Clean negatives (exhaustively searched)**: no differentiable / soft /
sigmoid / Gumbel / straight-through relaxation of the Gerber indicator exists;
no gradient-based learning of c; no decision-focused (task-loss) objective on
c; no state-conditioned c (one unpublished GitHub heuristic aside); the
DFL-for-covariance literature contains zero Gerber content. Patent family
(through US12567113B2) claims only a "predetermined magnitude" threshold.

**The prior art is corroboration, not just threat**: every attempt collapses c
to one global scalar found by zeroth-order search against a **Sharpe**
objective, and every one reporting OOS shows no gain. That is exactly the
failure mode our own Sharpe-objective ablation reproduces (c collapses to
0.075, OOS Sharpe 0.37 < variance-trained 0.47). We can explain the prior
literature's failure.

**Narrowest defensible novelty claim** (use verbatim in the paper):
> We give the first differentiable formulation of the Gerber statistic, in
> which the threshold coefficient is learned by gradient descent through a
> relaxed indicator against a decision-focused (portfolio-variance) objective,
> and the first in which it is conditioned on market state rather than fixed
> to a single global scalar. Prior work either holds the coefficient fixed
> while making the correlation dynamic (Algieri et al. 2021; Leccadito et al.
> 2024), reports sensitivity across a small fixed grid (Gerber et al. 2022;
> Flint & Polakow 2023), or selects one global scalar by black-box search on a
> Sharpe objective (Zhou 2024; Rubsamen 2023; Abu Khalaf & Smyth 2025) — none
> of which yields an out-of-sample improvement, a result our ablations explain.

**Citations that frame the gap as long-recognized**: Marakbi (KTH MSc 2016,
p.70) — "the threshold may be dynamically adjusted based on recent behavior of
the underlying variable, or alternatively, according to the prevailing market
regime. This calls for further research."; Smyth & Broby (FRL 2022) — c=0.5 is
"far from optimal for our enhanced statistic"; Dias (IME-USP MSc 2024) future
work — "executing the Gerber statistic considering an adaptative threshold c".

**Unresolved risks — must check before submission**:
1. **Smyth, Ernst & Miao (2025) "MINNs", SSRN 5336779** — highest risk. Two
   Gerber co-authors; learns "portfolio weights and interpretable covariance
   structure simultaneously"; cited by arXiv:2512.23021 as the vehicle for
   learning IQ parameters including c. SSRN-only, Cloudflare-gated — obtain
   via institutional access or email the author.
2. **Kim, Park, Shin & Bae (2026), J. Financial Data Science 8(1):31-47** —
   only peer-reviewed "modified Gerber statistic"; modification is LLM-derived
   semantic info, no threshold learning per abstract. Paywalled; residual risk.
3. Smyth's 2026 SSRN line (Kernel-IQ, Squeeze Kernel) — likely same
   benchmark-tuning pattern; do not assert either way in print.

### Final paper thesis after E-package

"When does it pay to learn what counts as co-movement?" — the threshold is
not a universal constant: it moves with time (0.7→1.6 post-COVID), flips
direction across asset classes (equities up, crypto down), and its path
depends on the decision layer. Learning it end-to-end pays exactly where
the decision is covariance-sensitive (long-short GMV, N>T: matches ANS
SOTA, avoids rank-deficiency blowups, robust to ridge/init/seed, portable
within asset class) and is unnecessary or harmful where the decision is
robust (long-only, small-N flat basin) — a complete, honest map.

### Updated paper storyline (evidence-backed)

1. Threshold choice matters and no fixed c is safe: optimum shifts with
   time (0.7→1.6), direction (asym 0.95/1.69), universe size (1.64 vs 0.89),
   and mis-setting is catastrophic in high dim (rank-deficiency blow-ups).
2. DiffGerber = differentiable framework that learns thresholds from the
   decision loss: matches best-fixed in easy regimes (N=100, flat basin),
   wins in hard regimes (N>T), never blows up.
3. T-state: state-conditioned co-movement definition — best where estimation
   is hardest; interpretable c_t trajectory (money figure).
4. Theory: PSD-by-construction G-cos family (+ new hard variant in the limit).
