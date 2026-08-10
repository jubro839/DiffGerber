"""Differentiable (soft) Gerber co-movement matrices."""

import torch
import torch.nn.functional as F
from torch import nn


def soft_exceedance(z, c, tau, c_down=None):
    """Three-state soft exceedance assignment via tempered softmax.

    z      : (..., T, N) volatility-normalized returns
    c      : upper thresholds (> 0), broadcastable to z
    c_down : lower thresholds; defaults to c (symmetric)
    tau    : temperature; as tau -> 0, states harden to indicators

    Returns (u, d, n), each (..., T, N), with u + d + n = 1:
        u -> 1{z > c}, d -> 1{z < -c_down}, n -> otherwise.
    """
    c = torch.as_tensor(c, dtype=z.dtype, device=z.device)
    cd = c if c_down is None else torch.as_tensor(c_down, dtype=z.dtype, device=z.device)
    logits = torch.stack(
        [(z - c) / tau, (-z - cd) / tau, torch.zeros_like(z)], dim=-1
    )
    p = F.softmax(logits, dim=-1)
    return p[..., 0], p[..., 1], p[..., 2]


def hard_ternary(z, c, c_down=None):
    """Hard Gerber ternary signal m in {-1, 0, +1}. Not differentiable."""
    c = torch.as_tensor(c, dtype=z.dtype, device=z.device)
    cd = c if c_down is None else torch.as_tensor(c_down, dtype=z.dtype, device=z.device)
    return (z > c).to(z.dtype) - (z < -cd).to(z.dtype)


def _normalize_weights(weights, T, like):
    if weights is None:
        w = torch.full((T,), 1.0 / T, dtype=like.dtype, device=like.device)
    else:
        w = torch.as_tensor(weights, dtype=like.dtype, device=like.device)
        w = w / w.sum(-1, keepdim=True)
    return w


def gerber_from_ternary(x, n=None, weights=None, normalization="cos", eps=1e-12):
    """Build a Gerber matrix from (soft or hard) ternary signals.

    x : (..., T, N) ternary signals in [-1, 1]
    n : (..., T, N) neutral probabilities (required for 'gs2022')
    normalization:
        'cos'       — Gram-normalized cosine Gerber; PSD by construction, unit diag
        'gs2022'    — (nc - nd) / (1 - <n_i n_j>); soft relaxation of JPM 2022
        'gs2019'    — (nc - nd) / sqrt((1 - <n_i^2>)(1 - <n_j^2>)). NOTE: in
                      the hard limit n^2 = n, so the denominator collapses to
                      sqrt(num_ii num_jj) and this is IDENTICAL to 'cos'
                      (verified to 3e-15); both coincide with the published
                      Gerber Statistic 1 (Gerber et al. 2019), already known
                      to be PSD. It differs from 'cos' only during soft
                      training. Kept for the ablation record; do not present
                      it as a new estimator.
    """
    T = x.shape[-2]
    w = _normalize_weights(weights, T, x)
    xw = x * w.unsqueeze(-1)
    num = torch.einsum("...ti,...tj->...ij", xw, x)  # Gram numerator, PSD

    if normalization == "cos":
        d = torch.diagonal(num, dim1=-2, dim2=-1).clamp_min(eps)
        G = num / torch.sqrt(d.unsqueeze(-1) * d.unsqueeze(-2))
        eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        return G * (1 - eye) + eye
    elif normalization == "gs2022":
        if n is None:
            raise ValueError("'gs2022' normalization needs neutral probs n")
        nw = n * w.unsqueeze(-1)
        nn_pair = torch.einsum("...ti,...tj->...ij", nw, n)
        den = 1.0 - nn_pair
        # The statistic is UNDEFINED when a pair is jointly neutral in every
        # period (denominator -> 0). Clamping to eps=1e-12 silently returned
        # 0/1e-12 = 0, so an asset that never crossed its threshold got
        # G_ii = 0 -> a fake zero-variance asset that GMV loads ~100% onto.
        # Convention: treat those entries as uncorrelated, unit variance.
        tol = 0.5 / T                       # less than half an exceedance
        bad = den <= tol
        G = torch.where(bad, torch.zeros_like(num), num / den.clamp_min(tol))
        eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        diag = torch.diagonal(G, dim1=-2, dim2=-1)
        G = G + eye * (diag < tol).to(G.dtype).unsqueeze(-1)
        # The published (hard) statistic has a unit diagonal; the soft
        # relaxation does not (numerator uses (u-d)^2, denominator u+d), which
        # deflated variances by up to 56% at tau=0.2 in psd_clip=False runs.
        # Renormalizing to a correlation matrix restores the hard-limit
        # semantics at every temperature and preserves PSD.
        d = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(tol)
        return G / torch.sqrt(d.unsqueeze(-1) * d.unsqueeze(-2))
    elif normalization == "gs2019":
        if n is None:
            raise ValueError("'gs2019' normalization needs neutral probs n")
        d = (1.0 - (w.unsqueeze(-1) * n * n).sum(-2)).clamp_min(eps)
        return num / torch.sqrt(d.unsqueeze(-1) * d.unsqueeze(-2))
    raise ValueError(f"unknown normalization: {normalization}")


def psd_project(G, floor=1e-6):
    """Differentiable eigenvalue clip + re-normalization to unit diagonal."""
    evals, evecs = torch.linalg.eigh(G)
    evals = evals.clamp_min(floor)
    G = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
    d = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(floor)
    return G / torch.sqrt(d.unsqueeze(-1) * d.unsqueeze(-2))


class SoftGerber(nn.Module):
    """Differentiable Gerber co-movement matrix.

    forward(z, c, tau) -> G with z (..., T, N), c broadcastable thresholds.
    """

    def __init__(self, normalization="cos", psd_clip=None, eps=1e-12,
                 straight_through=False, denominator_grad=True):
        """psd_clip is accepted for backward compatibility and IGNORED: the
        old `psd_project` never clipped anything (soft GS2022 min-eigenvalue
        was >= 0.033 in every configuration tested), it only renormalized the
        diagonal — which `gerber_from_ternary` now does unconditionally.

        straight_through=True makes the FORWARD pass exactly the hard Gerber
        statistic while keeping the soft gradient, so the object being trained
        is the object being deployed (otherwise they differ by ~19-31% of
        gross notional in GMV weights even at tau=0.02).

        With straight_through, BOTH ternary channels get the treatment: the
        signal x (the numerator's factor) and the neutral probability n (the
        GS2022 denominator's factor), so gradients flow through the numerator
        AND the denominator while the forward pass evaluates the hard values
        of both. denominator_grad=False severs the n path (n is then the hard
        indicator with no soft gradient) — an ablation switch to measure what
        the denominator's gradient contributes, never a production setting.
        """
        super().__init__()
        self.normalization = normalization
        self.eps = eps
        self.straight_through = straight_through
        self.denominator_grad = denominator_grad

    def forward(self, z, c, tau, weights=None, c_down=None):
        u, d, n = soft_exceedance(z, c, tau, c_down=c_down)
        x = u - d
        if self.straight_through:
            x_hard = hard_ternary(z, c, c_down=c_down)
            n_hard = (x_hard == 0).to(z.dtype)
            x = x + (x_hard - x).detach()
            n = (n + (n_hard - n).detach() if self.denominator_grad
                 else n_hard)
        elif not self.denominator_grad:
            n = n.detach()
        return gerber_from_ternary(
            x, n=n, weights=weights, normalization=self.normalization, eps=self.eps
        )


def hard_gerber(z, c, normalization="cos", weights=None, c_down=None):
    """Reference hard Gerber matrix (no gradients through thresholds)."""
    m = hard_ternary(z, c, c_down=c_down)
    n = (m == 0).to(z.dtype)
    return gerber_from_ternary(m, n=n, weights=weights, normalization=normalization)
