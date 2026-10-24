"""Independent CPU FP64 decoder mathematics with explicit activation boundaries."""

import torch

from attention_reference import attention_reference, rope_reference
from block_reference import decode, gated_mlp


def norm_reference(x, weight, epsilon):
    source = x.cpu().double()
    return (
        source
        * (source.square().mean(-1, keepdim=True) + epsilon).rsqrt()
        * weight.cpu().double()
    ).to(x.dtype)


def linear_reference(x, weight):
    return (x.cpu().double() @ decode(weight).T).to(x.dtype)


def decoder_reference(weights, dims, x, cos, sin, *, prefix=None, epsilon=1e-6):
    norm = norm_reference(x, weights.attention_norm, epsilon)
    q = linear_reference(norm, weights.query).view(
        x.shape[0], dims.query_heads, dims.head_dim
    )
    k = linear_reference(norm, weights.key).view(
        x.shape[0], dims.kv_heads, dims.head_dim
    )
    v = linear_reference(norm, weights.value).view_as(k)
    q = rope_reference(q, cos, sin, dims.rotary_dim, dims.rope_layout)
    k = rope_reference(k, cos, sin, dims.rotary_dim, dims.rope_layout)
    if prefix is not None:
        k = torch.cat((prefix[0], k))
        v = torch.cat((prefix[1], v))
    attention = attention_reference(q, k, v, dims.head_dim**-0.5)
    projected = linear_reference(
        attention.reshape(x.shape[0], dims.query_width), weights.output
    )
    residual = (x.cpu().double() + projected.double()).to(x.dtype)
    return gated_mlp(weights.mlp, residual, epsilon), (k, v)
