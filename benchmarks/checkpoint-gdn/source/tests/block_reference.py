"""Independent FP64 reference with explicit per-operator activation casts."""

import torch


def decode(weight):
    if hasattr(weight, "data"):
        return weight.data.cpu().double()
    codes, scales = weight.codes.cpu(), weight.scales.cpu()
    k = weight.in_features
    # Decode by bit positions, independently of the production codec.
    cols = torch.arange(k)
    values = (codes[:, cols // 2].to(torch.int32) >> ((cols % 2) * 4)) & 15
    signed = values - (values >= 8).to(torch.int32) * 16
    return signed.double() * scales[:, cols // 64].double()


def gated_mlp(weights, x, epsilon):
    dtype = x.dtype
    source = x.cpu().double()
    norm = source * (source.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    norm = (norm * weights.norm.cpu().double()).to(dtype).double()
    gate = (norm @ decode(weights.gate).T).to(dtype).double()
    up = (norm @ decode(weights.up).T).to(dtype).double()
    hidden = (gate * torch.sigmoid(gate) * up).to(dtype).double()
    down = (hidden @ decode(weights.down).T).to(dtype).double()
    return (source + down).to(dtype)
