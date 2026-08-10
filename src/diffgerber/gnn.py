"""Axis 2 prototype: soft-Gerber matrix as a learnable GNN adjacency."""

import torch
from torch import nn


def gerber_adjacency(G, mode="abs"):
    """Turn a soft Gerber matrix into a row-normalized adjacency with self-loops.

    Differentiable in the threshold parameters through G.
    """
    A = G.abs() if mode == "abs" else G.clamp_min(0.0)
    eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    A = A * (1 - eye) + eye
    return A / A.sum(-1, keepdim=True).clamp_min(1e-12)


class GerberGCNLayer(nn.Module):
    """One propagation step H' = act(A H W) over the Gerber graph."""

    def __init__(self, in_dim, out_dim, act=nn.ReLU()):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = act

    def forward(self, H, A):
        return self.act(self.lin(torch.einsum("...ij,...jf->...if", A, H)))


class GerberGNNForecaster(nn.Module):
    """Minimal asset-level forecaster on the Gerber graph (Axis 2 skeleton)."""

    def __init__(self, in_dim, hidden=32, layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(
            GerberGCNLayer(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, node_feats, G):
        A = gerber_adjacency(G)
        H = node_feats
        for layer in self.layers:
            H = layer(H, A)
        return self.head(H).squeeze(-1)
