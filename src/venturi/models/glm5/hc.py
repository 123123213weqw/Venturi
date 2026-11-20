"""GLM-5.3-Flash mHC (manifold-constrained Hyper-Connections).

Multi-stream residual mixing: computes pre/post/comb coefficients from the
flattened stream state, applies Sinkhorn-Knopp to the comb matrix to
approximately satisfy a doubly-stochastic constraint, collapses streams for
the sub-layer input, and re-injects the block output with post/comb.
HC parameters (fn/base/scale) are kept in float32 per the transformers
_keep_in_fp32_modules_strict convention.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Glm5HyperConnection(nn.Module):
    """mHC: Sinkhorn-mixed multi-stream residual for GLM-5.3-Flash."""

    def __init__(self, streams, hidden_size, sinkhorn_iterations=20, eps=1e-6):
        super().__init__()
        self.streams = streams
        self.hidden_size = hidden_size
        self.sinkhorn_iterations = sinkhorn_iterations
        self.eps = eps
        # (2+N)*N coefficients: pre N + post N + comb N*N
        total = (2 + streams) * streams
        self.fn = nn.Parameter(torch.randn(total, streams * hidden_size) * 0.02)
        self.base = nn.Parameter(torch.zeros(total))
        self.scale = nn.Parameter(torch.ones(3))
        # Unweighted RMSNorm (no learnable weight)
        self.norm_eps = eps

    def _unweighted_rms(self, x):
        return x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, hidden_streams):
        """Compute (post, comb, collapsed) from streams [B,T,N,D].

        Returns:
            post: [B,T,N] — post-sublayer scaling coefficients in (0,2)
            comb: [B,T,N,N] — approximately doubly-stochastic mixing matrix
            collapsed: [B,T,D] — weighted sum of streams for the sub-layer
        """
        batch, length, streams, dim = hidden_streams.shape
        dtype = hidden_streams.dtype

        flat = hidden_streams.reshape(batch * length, streams * dim)
        normed = self._unweighted_rms(flat)
        # FP32 projection
        coefficients = F.linear(normed, self.fn.float()) + self.base.float()

        n = self.streams
        pre_w, post_w, comb_w = coefficients.split([n, n, n * n], dim=-1)
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        # Activations (transformers: sigmoid + eps for pre, 2*sigmoid for post, softmax + eps for comb)
        pre = torch.sigmoid(pre_w * pre_scale) + self.eps
        post = 2 * torch.sigmoid(post_w * post_scale)
        # Reshape comb to [B*T, N, N] BEFORE Sinkhorn (dim=-2/-1 need the matrix form)
        comb = torch.softmax(comb_w * comb_scale, dim=-1) + self.eps
        comb = comb.view(batch * length, n, n)

        # Sinkhorn-Knopp: column first, then alternate row/column
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iterations - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)

        pre = pre.view(batch, length, n)
        post = post.view(batch, length, n)
        comb = comb.view(batch, length, n, n)

        # Collapse: weighted sum of streams
        collapsed = (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=2).to(dtype)

        return post, comb, collapsed

    def inject(self, block_output, residual_streams, post, comb):
        """Re-expand block output and mix with residual streams.

        Returns new streams [B,T,N,D].
        """
        dtype = residual_streams.dtype
        # post [B,T,N] × block [B,T,D] → [B,T,N,D]
        expanded = post.to(dtype).unsqueeze(-1) * block_output.to(dtype).unsqueeze(-2)
        # comb.T @ residual: [B,T,N,N] @ [B,T,N,D] → [B,T,N,D]
        mixed = torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual_streams)
        return expanded + mixed
