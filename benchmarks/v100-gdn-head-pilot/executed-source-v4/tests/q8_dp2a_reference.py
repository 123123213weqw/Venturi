"""Independent CPU reconstruction, not a device-kernel or model implementation."""

import torch


def quantize_reference(x):
    assert x.dtype == torch.float16 and x.device.type == "cpu" and x.ndim == 2
    t, k = x.shape
    groups = (k + 31) // 32
    padded = torch.zeros(t, groups * 32, dtype=torch.float32)
    padded[:, :k] = x.float()
    values = padded.reshape(t, groups, 32)
    valid = torch.isfinite(values).all(-1)
    maximum = values.double().abs().amax(-1)
    scales = (maximum / 32767).float()
    inverse = torch.where(maximum > 0, 32767 / maximum, 0).float()
    codes = torch.round((values * inverse[:, :, None]).float()).clamp(-32767, 32767)
    codes[~valid] = 0
    scales[~valid] = float("nan")
    codes = codes.to(torch.int16)
    represented = (codes.double() * scales.double()[:, :, None]).reshape(t, -1)[:, :k]
    return codes, scales, represented


def projection_reference(x, codes, scales):
    _, _, represented = quantize_reference(x)
    w = codes.double() * scales.double().repeat_interleave(32, -1)
    return represented @ w[:, : x.shape[1]].T
