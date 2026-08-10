"""Threshold modules: global, per-asset, and market-state-conditioned."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def inv_softplus(y):
    return math.log(math.expm1(y))


class GlobalThreshold(nn.Module):
    """Single learnable scalar threshold c = softplus(theta), init at `init`."""

    def __init__(self, init=0.5):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(inv_softplus(init)))

    def forward(self, feats=None):
        return F.softplus(self.raw)


class PerAssetThreshold(nn.Module):
    """Per-asset thresholds c_i = softplus(theta_i), init at `init`."""

    def __init__(self, n_assets, init=0.5):
        super().__init__()
        self.raw = nn.Parameter(torch.full((n_assets,), inv_softplus(init)))

    def forward(self, feats=None):
        return F.softplus(self.raw)


class AsymThreshold(nn.Module):
    """Separate learnable up/down thresholds (c_up, c_down), init symmetric."""

    def __init__(self, init=0.5):
        super().__init__()
        self.raw_up = nn.Parameter(torch.tensor(inv_softplus(init)))
        self.raw_down = nn.Parameter(torch.tensor(inv_softplus(init)))

    def forward(self, feats=None):
        return F.softplus(self.raw_up), F.softplus(self.raw_down)


class VocabAssetThreshold(nn.Module):
    """Per-asset thresholds over a fixed symbol vocabulary; folds with
    different universes gather their slice by index."""

    def __init__(self, vocab, init=0.5):
        super().__init__()
        self.vocab = {s: i for i, s in enumerate(vocab)}
        self.raw = nn.Parameter(torch.full((len(vocab),), inv_softplus(init)))

    def forward(self, symbols):
        idx = torch.tensor([self.vocab[s] for s in symbols])
        return F.softplus(self.raw[idx])


class StateThresholdNet(nn.Module):
    """MLP mapping market-state features to per-time thresholds.

    feats (..., T, F) -> c (..., T, out_dim), out_dim = n_assets or 1.
    c = c_min + softplus(MLP(feats)). The final layer starts at zero weights
    with bias set so c == init everywhere: training begins exactly at the
    published fixed-threshold estimator.
    """

    def __init__(self, n_features, n_assets=None, hidden=32, init=0.5, c_min=0.05,
                 c_max=3.0):
        super().__init__()
        out_dim = n_assets if n_assets is not None else 1
        self.c_min = c_min
        # Unbounded thresholds let the net drive c up to 4.17 in an earlier
        # run, pushing whole windows below the exceedance threshold and
        # degenerating the estimator. Cap smoothly.
        self.c_max = c_max
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.head.weight)
        # invert the tanh squash so forward() returns EXACTLY `init` at step 0
        span = c_max - c_min
        raw0 = span * math.atanh((init - c_min) / span)
        nn.init.constant_(self.head.bias, inv_softplus(raw0))

    def forward(self, feats):
        raw = F.softplus(self.head(self.body(feats)))
        span = self.c_max - self.c_min
        return self.c_min + span * torch.tanh(raw / span)
