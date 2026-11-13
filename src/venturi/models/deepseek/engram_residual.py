"""CPU Engram gated-residual oracle, NOT lookup/hash/projection or a model.

Inputs must already be independently bound/dequantized/projected. No native
model path is redirected here; this does not enable DeepSeek inference or claim
IQ4/FP8 equivalence. The normalization is per residual copy, unlike mHC's joint
stream normalization. Engram lookup is not speculative decoding.
"""
import math

import torch


def reference_engram_residual(stream, key, value, query_weight, key_weight, *,
                              norm_epsilon, signed_sqrt_floor=1e-6, token_mask=None):
    """[B,T,H,D] stream/key, [B,T,D] shared value, [H,D] weights.

    FP32 gate arithmetic and a signed, floored square root precede sigmoid.
    Norm epsilon and the sqrt floor are separate constants. Return stream dtype.
    """
    for parameter in (norm_epsilon, signed_sqrt_floor):
        if (isinstance(parameter,bool) or not isinstance(parameter,(int,float))
                or not math.isfinite(parameter) or parameter<=0):
            raise ValueError('Finite positive normalization epsilon/sqrt floor required')
    for tensor in (stream,key,value,query_weight,key_weight):
        if (not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu'
                or tensor.dtype not in (torch.float32,torch.float16,torch.bfloat16)
                or not torch.isfinite(tensor).all()):
            raise ValueError('Finite supported floating CPU oracle tensors required')
    if (stream.ndim!=4 or any(n==0 for n in stream.shape) or key.shape!=stream.shape
            or value.shape!=stream.shape[:2]+stream.shape[-1:]
            or query_weight.shape!=stream.shape[-2:] or key_weight.shape!=stream.shape[-2:]):
        raise ValueError('Matching nonempty Engram stream/key/value/weight shapes required')
    if token_mask is not None and (not isinstance(token_mask,torch.Tensor)
            or token_mask.device.type!='cpu' or token_mask.dtype!=torch.bool
            or token_mask.shape!=stream.shape[:2]):
        raise ValueError('Optional boolean CPU token mask must match [batch,tokens]')
    h,k=stream.float(),key.float()
    weights=query_weight.float()*key_weight.float()
    rstd=torch.rsqrt(h.square().mean(-1)+norm_epsilon)*torch.rsqrt(k.square().mean(-1)+norm_epsilon)
    dot=(h*weights*k).sum(-1)*rstd*stream.shape[-1]**-0.5
    gate=torch.sigmoid(torch.copysign(dot.abs().clamp_min(signed_sqrt_floor).sqrt(),dot))
    if token_mask is not None:gate=gate.masked_fill(~token_mask.unsqueeze(-1),0)
    result=(h+gate.unsqueeze(-1)*value.float().unsqueeze(-2)).to(stream.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite Engram residual arithmetic')
    return result
