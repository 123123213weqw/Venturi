"""Independent CPU reconstruction of the explicitly declared activation profile."""

import torch


def quantize_reference(x, passes):
    assert x.dtype == torch.float16 and x.device.type == "cpu"
    tokens, k = x.shape
    groups = (k + 31) // 32
    residual = torch.zeros(tokens, groups * 32, dtype=torch.float32)
    residual[:, :k] = x.float()
    residual = residual.reshape(tokens, groups, 32)
    codes, scales = [], []
    for _ in range(passes):
        maximum = residual.double().abs().amax(-1)
        scale = (maximum / 127).float()
        inverse = torch.where(maximum > 0, 127 / maximum, 0).float()
        q = torch.round((residual * inverse[:, :, None]).float()).clamp(-127, 127)
        # Double intermediate reproduces correctly rounded FP32 subtraction/FMA,
        # independently of the device intrinsic and packing implementation.
        residual = (residual.double() - q.double() * scale.double()[:, :, None]).float()
        codes.append(q.to(torch.int8))
        scales.append(scale)
    q, d = torch.stack(codes), torch.stack(scales)
    represented = (
        (q.double() * d.double()[:, :, :, None]).sum(0).reshape(tokens, -1)[:, :k]
    )
    return q, d, represented


def projection_reference(x, codes, scales, passes):
    _, _, represented = quantize_reference(x, passes)
    weight = codes.double() * scales.double().repeat_interleave(32, -1)
    return represented @ weight[:, : x.shape[1]].T
