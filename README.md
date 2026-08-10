# Learning the Gerber threshold with surrogate gradients

Anonymous code release accompanying the submission *Learning the Gerber
Threshold: Surrogate Gradients for Decision-Focused Covariance Estimation*.

The Gerber statistic counts co-movements that exceed a threshold `c·σ`, with
`c` fixed by convention at 0.5. Because the statistic is built from
indicators, the portfolio objective is piecewise constant in `c` and its exact
gradient is zero almost everywhere. This repository defines a tempered
three-state surrogate that is differentiable in `c`, takes gradients through
it while a straight-through construction keeps the deployed statistic
bit-exactly the published one, and optimizes the threshold against realized
portfolio variance through a differentiable minimum-variance layer.
`experiments/forward_backward_ablation.py` measures what that substitution
costs against a direct numerical gradient of the exact objective.

## What you can run, and what it needs

Daily prices and point-in-time index membership come from a licensed vendor
warehouse and **cannot be redistributed**, so reproduction is layered:

| Tier | Needs | What it reproduces |
|------|-------|--------------------|
| **A** | nothing beyond Python | Every table, figure and number in the paper, from the shipped result files |
| **B** | nothing beyond Python | The ground-truth simulation study end to end |
| **C** | a price/membership source | The full walk-forward experiments |

Tier A is the one to start with: `paper/verify_numbers.py` re-checks 161
claims from the manuscript against `experiments/results/` and exits non-zero
on any mismatch. It checks orderings as well as numbers, so a rerun that
reshuffles the panel fails rather than passing quietly.

## Setup

```bash
conda env create -f environment.yml && conda activate diffgerber
# or: pip install -r requirements.txt
```

Tiers A and B need only `torch`, `pandas`, `numpy`, `scipy`, `scikit-learn`
and `matplotlib`. `cvxpylayers` is used by the long-only decision layer and
`psycopg` only by the warehouse loader.

## Tier A — verify the paper without any data

```bash
python paper/verify_numbers.py     # assertions against experiments/results/
python paper/make_tables.py        # regenerates paper/tables/T1..T6 (LaTeX)
python paper/make_figures.py       # regenerates paper/figures/*.pdf
python tools/check_provenance.py   # every input traces to a producing script
```

No number in the manuscript is typed by hand: the tables are generated from
the CSV files in `experiments/results/`, and `verify_numbers.py` is the gate
that keeps text and data in agreement. `check_provenance.py` closes the other
direction — it walks from the paper's inputs back to the experiment script
that writes each one, so a result file that nothing in the release can
regenerate is reported as an orphan rather than passing unnoticed.

## Tier B — runs with no external data

```bash
python experiments/smoke_test.py       # ~30 s. Soft→hard consistency, PSD,
                                       # gradient flow, toy end-to-end training
python experiments/demo_synthetic.py   # ~15 s. Full pipeline (learning,
                                       # annual refits, walk-forward, metrics)
                                       # on synthetic heavy-tailed returns
python experiments/simulation_study.py # ~10 min. Section 6.7: the robustness
                                       # crossover and the two-target result
```

`demo_synthetic.py` is the quickest way to see the whole pipeline work. Its
performance table is a pipeline check rather than evidence — 25 assets and one
generator draw cannot separate estimators — but the learned threshold does
move below 0.5, which is the tail mechanism the paper describes.

## Tier C — full experiments (requires a data source)

`src/diffgerber/data.py` is the only module that touches the warehouse. It
reads four things from a PostgreSQL `marts` schema:

| Table | Columns used |
|-------|--------------|
| `mv_prices_adjusted` (+ `mst_security`) | `market_date`, `fmp_symbol`, `close_total_return` |
| `mst_universe_history` | `fmp_symbol`, `effective_from`, `effective_to` |
| `security_market_cap` | `security_id`, `market_date`, `market_cap_fmp` |
| `mst_universe` | `fmp_symbol`, `sector`, `cik` |

Crypto experiments additionally read `crypto_prices`, and the market-state
module reads index closes from `index_prices`. Any source that supplies daily
total-return prices, membership intervals with effective dates, daily market
capitalizations and sector labels can be substituted by reimplementing that
one module; nothing downstream is source-specific.

Credentials are read from a dotenv file pointed at by `DIFFGERBER_FMP_ENV`,
containing `FMP_DB_HOST`, `FMP_DB_PORT`, `FMP_DB_NAME`, `FMP_DB_USER`,
`FMP_DB_PASSWORD`. Query results are cached as parquet under `data/cache/`.

With a data source in place, the sequence behind the paper is:

| Script | Produces | Approx. runtime |
|--------|----------|-----------------|
| `headline_pack.py` | the headline panel: sector-balanced universes (the paper reports N = 30 and 70; the sweep also writes N = 20, 40, 50, 100), every arm and baseline, tests, sub-periods, certainty equivalents | 90 min |
| `main_experiments.py gs2022` | market-cap top-100 panel, the second design | 25 min |
| `threshold_grid.py` | oracle recovery, Figure 2 | 40 min |
| `e1_ans.py` | analytical nonlinear shrinkage baseline | 20 min |
| `top_universe_5metrics.py` | five-metric panels, DG-HRP / DG-Shrink | 45 min |
| `sector_universe.py` | sector-balanced universes | 40 min |
| `architecture_v2b.py` | rolling selection rules | 45 min |
| `arch_v3_dev.py` → `arch_v3_holdout.py` | development sweep, then the single held-out confirmation | 60 + 60 min |
| `economic_value.py` | cost curves, certainty equivalents, sub-periods | 25 min |
| `prereg_extensions.py` | turnover penalty, learned shrinkage intensity | 40 min |
| `reinforcement.py` | paired bootstrap, 2-D grid control, 3-parameter joint learning, estimation-objective contrast, analytic-δ baseline, mechanism table | 50 min |
| `forward_backward_ablation.py` | hard/soft forward x surrogate/finite-difference backward, with learned threshold paths and per-refit timings | 15 min |
| `relaxation_gap.py` | how far the soft matrix sits from the hard one, in matrix norm and in the decision, along the annealing path | 5 min |
| `gradient_decomposition.py` | how much of dL/dc flows through the Gerber denominator vs the numerator, per fold and temperature | 5 min |
| `ablation_c_delta.py` | factorial over learning the threshold c and the shrinkage intensity delta: each alone, fixed grids, both, neither | 60 min |
| `mixed_universe_experiment.py` | mixed universes: the sector-balanced equity selection plus a fixed nine-ETF sleeve (IWM EFA EEM AGG HYG TIP GLD VNQ TLT), single learned threshold against the constant-threshold ceiling | 15 min |
| `mixed_group_experiment.py` | Table 1 Panel B: asset-class group thresholds (c_eq, c_slv) alone and jointly with the shrinkage blend, plus the 4x4 constant (c_eq, c_slv) ceiling grid | 25 min |
| `sector_ceiling_experiment.py` | the constant-threshold ceiling for the sector-balanced universes: a ten-point grid through the same protocol, so the learned coefficient can be read against the best a fixed choice could have made with hindsight | 20 min |
| `mixed_arms_experiment.py` | the remaining Panel B arms on the mixed universes: DG-Asym (learned up/down thresholds) and DG-HRP (learned correlation into the same hierarchical rule as the HRP comparator) | 10 min |
| `multiasset_experiment.py` | nine- and twelve-asset ETF universes (equities, bonds, credit, TIPS, gold, REITs, commodities): learned threshold, constant-threshold ceiling, baselines | 10 min |
| `ew_and_wealth.py` | equal-weight baseline and cumulative wealth paths | 30 min |
| `decision_dependence.py`, `sharpe_objective.py`, `crypto_experiment.py`, `ablation_stability.py`, `conditioning_control.py`, `reviewer_gaps.py` | boundary conditions and robustness | 20–40 min each |

Runtimes are for a laptop CPU; the learned components have at most a few
hundred parameters, so training is not the bottleneck.

Training is seeded (`torch.manual_seed(0)`) and runs in float64, so repeated
runs of the same script reproduce their threshold paths. The one caveat is
that multi-threaded floating-point reductions are not bit-reproducible: an
independent retrain of the headline configuration reproduces eight of its ten
annual thresholds exactly and differs by at most 0.007 in the other two, which
moves annualized volatility by half a basis point. Table 5 of the paper
reports that comparison.

## Layout

```
src/diffgerber/
  soft_gerber.py      tempered three-state gates, straight-through estimator,
                      hard/soft Gerber matrices, degeneracy handling
  threshold_net.py    global / asymmetric / per-asset / state-conditioned modules
  portfolio_layer.py  differentiable GMV (closed form), long-only QP layer,
                      variance and downside decision losses
  backtest.py         walk-forward engine, five-metric reporting, cost model
  baselines.py        sample, EWMA, Ledoit-Wolf, analytical nonlinear shrinkage,
                      HRP (single/average/ward linkage), fixed-threshold Gerber
  stats.py            Diebold-Mariano with Newey-West, stationary bootstrap
  data.py             point-in-time universe construction and loaders
experiments/          one script per result; results/ holds every number used
paper/                table, figure and verification pipeline
```

## Protocol

All experiments share a protocol fixed in advance: 252-day estimation window,
21-day rebalancing, evaluation from 2017, 10 bp costs (with 0/25/50 bp
sensitivity), scale-relative ridge `1e-3` applied identically in training and
evaluation, annual expanding refits that exclude folds whose realization
horizon crosses into the evaluation year, a fresh optimizer at each refit,
Adam at learning rate 0.02 for 40 epochs, and simple-return metrics.
Significance uses Diebold-Mariano on per-fold realized variances with an
automatic Newey-West bandwidth and the Harvey-Leybourne-Newbold correction.

Two documents in `experiments/results/` record how the numbers came to be:
`audit_impact.md` (final results, and a side-by-side comparison against the
pre-audit versions that a corrective code audit invalidated) and
`phase4_interim.md` (the working record, including claims that were later
retracted). Scripts whose results are labelled pre-registered in the paper
were committed before they were executed.

## License

Code: MIT (see LICENSE). No market data is included in this repository.
