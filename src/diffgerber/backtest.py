"""Walk-forward backtest engine for covariance-driven GMV portfolios."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def gmv_from_cov(Sigma, ridge=1e-3):
    """Evaluation-side GMV. Must stay identical to
    portfolio_layer.gmv_closed_form (same scale-relative ridge)."""
    N = Sigma.shape[0]
    scale = np.trace(Sigma) / N
    A = Sigma + ridge * scale * np.eye(N)
    x = np.linalg.solve(A, np.ones(N))
    return x / x.sum()


@dataclass
class FoldResult:
    date: pd.Timestamp
    symbols: list
    weights: np.ndarray
    realized_var: float  # mean squared daily portfolio return over the fold
    gross_return: float
    turnover: float


@dataclass
class BacktestResult:
    name: str
    folds: list = field(default_factory=list)
    daily_returns: pd.Series = None

    def metrics(self, cost_bps=10.0, periods=252):
        """The return panel holds LOG returns, so convert to simple returns
        before compounding or computing a Sharpe ratio: compounding log
        returns with (1+r).cumprod() overstates drawdowns (~2pp here) and
        using a log-return mean understates Sharpe by ~sigma/2, which is not
        method-neutral (the penalty scales with each method's own vol)."""
        r_log = self.daily_returns.copy()
        cost = pd.Series(0.0, index=r_log.index)
        for f in self.folds:
            if f.date in cost.index:
                cost.loc[f.date] += f.turnover * cost_bps * 1e-4
        net_log = r_log - cost
        net = np.expm1(net_log)                       # simple returns
        ann_vol = net.std(ddof=1) * np.sqrt(periods)
        ann_ret = net.mean() * periods
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        # Sortino: downside deviation vs a zero target, full-sample denominator
        downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)) * np.sqrt(periods)
        sortino = ann_ret / downside if downside > 0 else np.nan
        curve = np.exp(net_log.cumsum())              # correct compounding
        mdd = (curve / curve.cummax() - 1).min()
        turnovers = [f.turnover for f in self.folds]
        return {
            "ann_vol_gross": np.expm1(r_log).std(ddof=1) * np.sqrt(periods),
            "ann_vol_net": ann_vol,
            "ann_ret_net": ann_ret,
            "sharpe_net": sharpe,
            "sortino_net": sortino,
            "max_drawdown": mdd,
            # fold 0 is a full position build, not a steady-state trade
            "avg_turnover": np.mean(turnovers[1:]) if len(turnovers) > 1 else turnovers[0],
            "avg_turnover_incl_build": np.mean(turnovers),
            "n_folds": len(self.folds),
        }

    def fold_losses(self):
        """Per-fold realized variance — the GMV decision loss, for DM tests."""
        return np.array([f.realized_var for f in self.folds])


def run_walkforward(
    name,
    returns,          # wide daily log-returns DataFrame (full panel)
    weight_fn,        # (R_window ndarray (T,N), symbols, as_of) -> weights (N,)
    universe_fn,      # (as_of) -> list of symbols
    start,
    end,
    window=252,
    step=21,
):
    """Strict walk-forward: at each rebalance date, weights are computed from
    the trailing `window` days only; realized over the next `step` days."""
    dates = returns.index
    start_i = dates.searchsorted(pd.Timestamp(start))
    end_i = dates.searchsorted(pd.Timestamp(end))
    result = BacktestResult(name=name)
    daily = []
    prev_w = None
    prev_syms = None

    # only full-length folds: a truncated final fold has ~2.6x the sampling
    # variance of the others and would enter the DM test with equal weight
    for k in range(start_i, end_i - step + 1, step):
        as_of = dates[k]
        symbols = universe_fn(as_of)
        Rw = returns[symbols].iloc[k - window : k]
        if Rw.isna().any().any():
            Rw = Rw.fillna(0.0)
        w = weight_fn(Rw.values, symbols, as_of)

        fut = returns[symbols].iloc[k : min(k + step, end_i)].fillna(0.0)
        port = fut.values @ w
        port_s = pd.Series(port, index=fut.index)

        if prev_w is None:
            turnover = np.abs(w).sum()
        else:
            prev = pd.Series(prev_w, index=prev_syms)
            cur = pd.Series(w, index=symbols)
            union = cur.index.union(prev.index)
            turnover = (cur.reindex(union, fill_value=0.0)
                        - prev.reindex(union, fill_value=0.0)).abs().sum()

        result.folds.append(FoldResult(
            date=as_of, symbols=symbols, weights=w,
            realized_var=float((port ** 2).mean()),
            gross_return=float(port.sum()),
            turnover=float(turnover),
        ))
        daily.append(port_s)
        prev_w, prev_syms = w, symbols

    result.daily_returns = pd.concat(daily)
    return result


def covariance_weight_fn(cov_fn, ridge=1e-3):
    def fn(R, symbols, as_of):
        return gmv_from_cov(cov_fn(R), ridge=ridge)
    return fn
