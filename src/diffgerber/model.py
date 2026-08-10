"""End-to-end DiffGerber GMV model."""

import torch
from torch import nn

from .portfolio_layer import gmv_closed_form
from .soft_gerber import SoftGerber


class DiffGerberGMV(nn.Module):
    """threshold module -> soft Gerber -> covariance -> GMV weights.

    Scale (per-asset vol) is supplied externally and treated as fixed:
    learning capacity sits in the co-movement structure only.
    """

    def __init__(self, threshold_module, normalization="cos", ridge=1e-4):
        super().__init__()
        self.thresholds = threshold_module
        self.gerber = SoftGerber(normalization=normalization)
        self.ridge = ridge

    def forward(self, returns, scales, tau, feats=None, weights=None):
        """returns (..., T, N); scales (..., N); feats optional (..., T, F)."""
        z = returns / scales.unsqueeze(-2)
        c = self.thresholds(feats) if feats is not None else self.thresholds()
        G = self.gerber(z, c, tau, weights=weights)
        Sigma = scales.unsqueeze(-1) * G * scales.unsqueeze(-2)
        w = gmv_closed_form(Sigma, ridge=self.ridge)
        return {"w": w, "G": G, "Sigma": Sigma, "c": c}


def tau_schedule(step, total_steps, tau_start=0.2, tau_end=0.02):
    """Cosine annealing of the softmax temperature."""
    import math

    frac = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * frac))
