"""Differentiable portfolio decision layers."""

import torch
from torch import nn


def gmv_closed_form(Sigma, ridge=1e-3):
    """Long-short global-minimum-variance weights via a linear solve.

    w = (Sigma + ridge*s I)^{-1} 1 / (1' (Sigma + ridge*s I)^{-1} 1),
    with s = mean(diag(Sigma)) so `ridge` is scale-free. This MUST match
    backtest.gmv_from_cov exactly: an earlier version used an absolute ridge
    here and a scale-relative one at evaluation, making training ~7,700x more
    regularized than deployment for daily returns.
    Differentiable through torch.linalg.solve. Sigma: (..., N, N).
    """
    N = Sigma.shape[-1]
    scale = torch.diagonal(Sigma, dim1=-2, dim2=-1).mean(-1)
    eye = torch.eye(N, dtype=Sigma.dtype, device=Sigma.device)
    ones = torch.ones(Sigma.shape[:-1] + (1,), dtype=Sigma.dtype, device=Sigma.device)
    x = torch.linalg.solve(Sigma + ridge * scale * eye, ones).squeeze(-1)
    return x / x.sum(-1, keepdim=True)


class LongOnlyGMV(nn.Module):
    """Long-only GMV via a cvxpylayers QP with Cholesky parameterization.

    minimize ||L w||^2  s.t.  sum(w) = 1, w >= 0, with L L' = Sigma + ridge I.
    Constructed lazily so the module imports without cvxpylayers installed.
    """

    def __init__(self, n_assets, ridge=1e-3):
        super().__init__()
        self.n_assets = n_assets
        self.ridge = ridge
        self._layer = None

    def _build(self):
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        n = self.n_assets
        w = cp.Variable(n)
        L = cp.Parameter((n, n))
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(L @ w)), [cp.sum(w) == 1, w >= 0]
        )
        self._layer = CvxpyLayer(prob, parameters=[L], variables=[w])

    def forward(self, Sigma):
        if self._layer is None:
            self._build()
        # scale-relative ridge, identical to gmv_closed_form / gmv_from_cov;
        # an absolute ridge here was ~3,000x heavier than the long-short path
        eye = torch.eye(self.n_assets, dtype=Sigma.dtype, device=Sigma.device)
        scale = torch.diagonal(Sigma, dim1=-2, dim2=-1).mean(-1)
        L = torch.linalg.cholesky(Sigma + self.ridge * scale * eye).transpose(-1, -2)
        (w,) = self._layer(L)
        return w


def downside_decision_loss(w, future_returns):
    """Realized downside second moment (Sortino-aligned): mean of squared
    NEGATIVE portfolio returns over the horizon. A second-moment objective,
    so it retains the learnability that the Sharpe objective lacks."""
    port = torch.einsum("...hn,...n->...h", future_returns, w)
    return (torch.clamp(port, max=0.0) ** 2).mean(-1).mean()


def decision_loss(w, future_returns, prev_w=None, turnover_penalty=0.0):
    """Realized OOS portfolio second moment (GMV objective) + turnover.

    w: (..., N); future_returns: (..., H, N).
    """
    port = torch.einsum("...hn,...n->...h", future_returns, w)
    loss = (port**2).mean(-1)
    if prev_w is not None and turnover_penalty > 0:
        loss = loss + turnover_penalty * (w - prev_w).abs().sum(-1)
    return loss.mean()
