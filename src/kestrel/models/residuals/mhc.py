"""Manifold-Constrained Hyper-Connections (mHC) with Sinkhorn-Knopp doubly-stochastic mixing."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def sinkhorn_knopp(logits: torch.Tensor, num_iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Project a square matrix onto the Birkhoff polytope (doubly-stochastic, `||M|| <= 1`)."""
    mat = torch.exp(logits.float() - logits.float().amax(dim=-1, keepdim=True))
    for _ in range(num_iters):
        mat = mat / (mat.sum(dim=-1, keepdim=True) + eps)
        mat = mat / (mat.sum(dim=-2, keepdim=True) + eps)
    return mat


class ManifoldHyperConnection(nn.Module):
    """Multi-stream residual wrapper (`S=4`) with doubly-stochastic stream mixing."""

    def __init__(self, num_streams: int = 4, sinkhorn_iters: int = 10) -> None:
        super().__init__()
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Initialize near identity so early training behaves like standard residual connections
        init_res = torch.full((num_streams, num_streams), -4.0)
        init_res.fill_diagonal_(0.0)
        self.h_res_logits = nn.Parameter(init_res)
        self.h_pre_logits = nn.Parameter(torch.zeros(num_streams))
        self.h_post_logits = nn.Parameter(torch.zeros(num_streams))

    @staticmethod
    def expand(x: torch.Tensor, num_streams: int) -> torch.Tensor:
        """Expand `(B, T, D)` into `(B, T, S, D)` at model input."""
        return x.unsqueeze(2).expand(-1, -1, num_streams, -1).contiguous()

    @staticmethod
    def collapse(h: torch.Tensor) -> torch.Tensor:
        """Collapse `(B, T, S, D)` back to `(B, T, D)` by averaging streams."""
        return h.mean(dim=2)

    def mixing_matrix(self) -> torch.Tensor:
        """Return the `(S, S)` doubly-stochastic residual mixing matrix."""
        return sinkhorn_knopp(self.h_res_logits, num_iters=self.sinkhorn_iters)

    def pre_aggregate(self, h: torch.Tensor) -> torch.Tensor:
        """Aggregate `S` streams `(B, T, S, D)` into a single `(B, T, D)` sublayer input."""
        w_pre = torch.softmax(self.h_pre_logits.float(), dim=0).to(h.dtype)
        return torch.einsum("s,btsd->btd", w_pre, h)

    def post_update(self, h: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """Apply doubly-stochastic residual mix + weighted `sublayer_out` `(B, T, D)`."""
        m_res = self.mixing_matrix().to(h.dtype)
        w_post = (2.0 * torch.sigmoid(self.h_post_logits.float())).to(h.dtype)
        mixed = torch.einsum("ij,btjd->btid", m_res, h)
        added = torch.einsum("s,btd->btsd", w_post, sublayer_out)
        return mixed + added

    def forward(
        self,
        h: torch.Tensor,
        sublayer_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run `sublayer_fn` on aggregated streams and update all `S` streams."""
        x_in = self.pre_aggregate(h)
        out = sublayer_fn(x_in)
        return self.post_update(h, out)
