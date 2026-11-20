"""Experimental W5A16 RTN codec for the q5_g64_fp16 row-split plane layout.

This is NOT AWQ/GPTQ or a quality-qualified model. The codec mirrors the
Q4 family: CPU weight-only RTN, FP16 scale first, ties-to-even, q in
[-16, 15] over contiguous K64 groups, with the fifth code bit stored in a
separate high plane exactly as described by q5_g64_fp16. No import-time
compilation, production/default binding or decoded-weights cache.
"""

import torch
from torch.nn import functional as F

from venturi.ops.weights import Q5RowSplitWeight


CODEC = 'rtn_symmetric15_g64_fp16_row_split_k128_v1'


def quantize_q5_rtn(weight, *, max_elements=16*2**20):
    """CPU weight-only RTN: FP16 scale first, ties-to-even, q in [-16,15].

    Each contiguous K64 group uses max(abs(w))/15. Scale is rounded to FP16
    BEFORE code rounding and floored at the smallest positive FP16 subnormal.
    Padding is zero. Output has independent owned storage.
    """
    if not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.ndim != 2 or (
            weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or min(weight.shape) < 1):
        raise ValueError('Explicit nonempty CPU floating weight required')
    if type(max_elements) is not int or max_elements <= 0 or weight.numel() > max_elements:
        raise ValueError('Q5 quantization exceeds explicit element budget')
    value = weight.float()
    if not torch.isfinite(value).all():
        raise ValueError('Finite source weights required')
    rows, cols = weight.shape
    padded = (cols+127)//128*128
    blocks = F.pad(value, (0, padded-cols)).reshape(rows, padded//64, 64)
    scales = (blocks.abs().amax(-1)/15).clamp_min(2**-24).half()
    if not torch.isfinite(scales).all():
        raise ValueError('Source scale overflows FP16')
    codes = (blocks/scales.float().unsqueeze(-1)).round().clamp(-16, 16).to(torch.int8)
    # clamp(-16,16) can emit +16 only when scale rounding moved the boundary;
    # saturate those to +15 so the stored code range stays exactly [-16, 15].
    codes = torch.where(codes == 16, torch.full_like(codes, 15), codes)
    if not torch.isfinite((codes.float()*scales.float().unsqueeze(-1)).half()).all():
        raise ValueError('Quantized weight overflows FP16 execution')
    # Fully padded groups may have scale zero; their codes remain exactly zero.
    return Q5RowSplitWeight.from_codes(codes.reshape(rows, padded)[:, :cols],
                                       scales[:, :(cols+63)//64])


def decode_q5_fp16(weight):
    """Bounded diagnostic/fallback FP16 matrix, no retained decoded cache."""
    weight.validate()
    unsigned = torch.stack((weight.codes & 15, weight.codes >> 4), -1).reshape(
        weight.out_features, weight.padded_k).to(torch.int16)
    bytes_ = weight.high.view(weight.out_features, -1, 8).to(torch.int16)
    bits = (bytes_.unsqueeze(-1) >> torch.arange(8, dtype=torch.int16)) & 1
    fifth = bits.reshape(weight.out_features, -1, 64)
    lanes = unsigned.view(weight.out_features, -1, 64) | (fifth << 4)
    signed = torch.where(lanes < 16, lanes, lanes-32)
    return (signed.float()*weight.scales.float().unsqueeze(-1)).reshape(
        weight.out_features, -1)[:, :weight.in_features].half()
