"""Phase 3 pilot: does a decision-learned Gerber threshold beat fixed c=0.5
on real point-in-time S&P large-cap data?

Protocol
- Universe: point-in-time top-100 S&P members by market cap at each rebalance.
- Walk-forward: window 252, rebalance every 21 trading days, 2016..2026-07.
- DiffGerber T-global: annual expanding refit — thresholds used in year Y are
  trained only on folds ending before Y (strict no look-ahead). Warm-started
  from the previous year's threshold. Evaluation starts 2017 (2016 is the
  first training year).
- Baselines evaluated on the same 2017+ fold set.
"""

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
from diffgerber.threshold_net import GlobalThreshold, inv_softplus

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

START_DATA, END_DATA = "2015-01-01", "2026-07-28"
WINDOW, STEP, TOP_N = 252, 21, 100
EVAL_START_YEAR = 2017
EPOCHS, LR = 40, 0.02
NORM = sys.argv[1] if len(sys.argv) > 1 else "cos"  # 'cos' or 'gs2022'
print(f"normalization: {NORM}")

print("loading data...")
mem = data.sp500_membership()
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return(START_DATA, END_DATA, symbols=all_syms)
rets = data.log_returns(px)
dates = rets.index

_uni_memo = {}
def universe_fn(as_of):
    if as_of not in _uni_memo:
        _uni_memo[as_of] = data.pit_universe(as_of, TOP_N, px, mem)
    return _uni_memo[as_of]

# rebalance calendar: every STEP days once WINDOW days of history exist
reb_idx = list(range(WINDOW, len(dates) - 1, STEP))
reb_dates = [dates[i] for i in reb_idx]
print(f"rebalances: {len(reb_dates)} ({reb_dates[0].date()} .. {reb_dates[-1].date()})")

# ---- precompute fold tensors once ----
print("preparing folds...")
folds = []  # (date, symbols, R_window tensor, R_future tensor)
for i in reb_idx:
    as_of = dates[i]
    syms = universe_fn(as_of)
    Rw = torch.as_tensor(rets[syms].iloc[i - WINDOW : i].fillna(0.0).values)
    Rf = torch.as_tensor(rets[syms].iloc[i : i + STEP].fillna(0.0).values)
    folds.append((as_of, syms, Rw, Rf))

# ---- DiffGerber T-global with annual expanding refit ----
def train_threshold(train_folds, init_c, epochs=EPOCHS, lr=LR):
    th = GlobalThreshold(init=init_c)
    layer = SoftGerber(NORM, straight_through=True)
    opt = torch.optim.Adam(th.parameters(), lr=lr)
    for ep in range(epochs):
        tau = tau_schedule(ep, epochs, 0.2, 0.02)
        opt.zero_grad()
        loss = 0.0
        for _, _, Rw, Rf in train_folds:
            s = Rw.std(0, unbiased=True).clamp_min(1e-8)
            G = layer(Rw / s, th(), tau)
            Sigma = s.unsqueeze(-1) * G * s.unsqueeze(-2)
            w = gmv_closed_form(Sigma)
            loss = loss + decision_loss(w, Rf)
        (loss / len(train_folds)).backward()
        opt.step()
    return float(th().detach())

print("training thresholds (annual expanding refit)...")
learned_c = {}  # year -> threshold trained on folds before that year
c_prev = 0.5
for year in range(EVAL_START_YEAR, 2027):
    train = [f for f in folds if f[0] < pd.Timestamp(f"{year}-01-01")]
    t0 = time.time()
    c_year = train_threshold(train, init_c=c_prev)
    learned_c[year] = c_year
    print(f"  {year}: c={c_year:.3f}  (trained on {len(train)} folds, "
          f"{time.time() - t0:.0f}s)")
    c_prev = c_year

# ---- evaluation on 2017+ folds ----
eval_folds = [f for f in folds if f[0].year >= EVAL_START_YEAR]
eval_start, eval_end = eval_folds[0][0], dates[-1]

def diffgerber_weight_fn(R, symbols, as_of):
    c = learned_c[as_of.year]
    Rt = torch.as_tensor(R)
    s = Rt.std(0, unbiased=True).clamp_min(1e-8)
    G = hard_gerber(Rt / s, c, NORM)
    Sigma = (s.unsqueeze(-1) * G * s.unsqueeze(-2)).numpy()
    return backtest.gmv_from_cov(Sigma)

runs = {}
contenders = {
    "sample": backtest.covariance_weight_fn(baselines.sample_cov),
    "ewma_hl126": backtest.covariance_weight_fn(baselines.ewma_cov),
    "ledoit_wolf": backtest.covariance_weight_fn(baselines.ledoit_wolf_cov),
    "gerber_cos_c0.5": backtest.covariance_weight_fn(
        lambda R: baselines.gerber_cov(R, 0.5, "cos")),
    "gerber_2022_c0.5": backtest.covariance_weight_fn(
        lambda R: baselines.gerber_cov(R, 0.5, "gs2022")),
    "hrp": lambda R, s, d: baselines.hrp_weights(R),
    "diffgerber_Tglobal": diffgerber_weight_fn,
}
print("running walk-forward evaluations...")
for name, wfn in contenders.items():
    t0 = time.time()
    runs[name] = backtest.run_walkforward(
        name, rets, wfn, universe_fn, eval_start, eval_end,
        window=WINDOW, step=STEP)
    m = runs[name].metrics(cost_bps=10)
    print(f"  {name:20s} vol_net={m['ann_vol_net']*100:6.2f}%  "
          f"sharpe={m['sharpe_net']:+.2f}  mdd={m['max_drawdown']*100:6.1f}%  "
          f"to={m['avg_turnover']:.2f}  ({time.time() - t0:.0f}s)")

print("\n== Diebold-Mariano on per-fold realized variance ==")
base = runs["diffgerber_Tglobal"].fold_losses()
for name in ["gerber_cos_c0.5", "gerber_2022_c0.5", "sample", "ledoit_wolf", "ewma_hl126"]:
    dm, p = stats.diebold_mariano(base, runs[name].fold_losses())
    print(f"  diffgerber vs {name:18s}: DM={dm:+.2f} p={p:.4f} "
          f"({'diffgerber better' if dm < 0 else 'baseline better'})")

print("\nlearned thresholds by year:", {y: round(c, 3) for y, c in learned_c.items()})
