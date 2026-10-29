"""Bounded CPU dequantized-F32 Engram projection/gate oracle, NOT native execution.

Inputs are explicit prepared rows and independently authenticated/dequantized
weights. Does not open checkpoints, choose precision, emulate BF16/FP8/fused GPU
rounding, or load a model. Its numerical reference uses an explicit F32 matmul.
"""
import torch

from venturi.models.engram_residual import reference_engram_residual


@torch.no_grad()
def reference_engram_from_rows(stream, rows, wkv_weight, query_weight, key_weight, *,
                                norm_epsilon, signed_sqrt_floor=1e-6, token_mask=None,
                                max_weight_values=2**20, max_work_values=2**20):
    """[B,T,C,E] rows -> F32 [(H+1)*D,C*E] linear -> key/value gate.

    Stream [B,T,H,D] keeps its original dtype. Copy/head order is explicit:
    flatten ONLY the row/head-embedding axes, key occupies the first H*D output
    columns and the single shared value the last D. No in-place input changes.
    Default budgets are tiny-oracle limits, not a claim to load real WKV weights.
    """
    if (type(max_weight_values) is not int or not 0<max_weight_values<=2**28
            or type(max_work_values) is not int or not 0<max_work_values<=2**20):
        raise ValueError('Explicit bounded projection weight/work budgets required')
    if (not isinstance(stream,torch.Tensor) or stream.device.type!='cpu'
            or stream.dtype not in (torch.float32,torch.float16,torch.bfloat16)
            or stream.ndim!=4 or any(n==0 for n in stream.shape)):
        raise ValueError('Nonempty supported floating CPU [B,T,H,D] stream required')
    if (not isinstance(rows,torch.Tensor) or rows.device.type!='cpu' or rows.dtype!=torch.float32
            or rows.ndim!=4 or any(n==0 for n in rows.shape) or rows.shape[:2]!=stream.shape[:2]):
        raise ValueError('Matching nonempty F32 CPU [B,T,columns,head_dim] rows required')
    batch,tokens,copies,dim=stream.shape
    inputs=rows.shape[2]*rows.shape[3];outputs=(copies+1)*dim
    if (not isinstance(wkv_weight,torch.Tensor) or wkv_weight.device.type!='cpu'
            or wkv_weight.dtype!=torch.float32 or wkv_weight.shape!=(outputs,inputs)
            or wkv_weight.numel()>max_weight_values
            or rows.numel()>max_work_values or batch*tokens*outputs>max_work_values):
        raise ValueError('F32 CPU [(copies+1)*hidden,columns*head_dim] WKV within budgets required')
    for weight in (query_weight,key_weight):
        if (not isinstance(weight,torch.Tensor) or weight.device.type!='cpu'
                or weight.dtype!=torch.float32 or weight.shape!=(copies,dim)):
            raise ValueError('Explicit F32 CPU query/key weights [copies,hidden] required')
    if token_mask is not None and (not isinstance(token_mask,torch.Tensor)
            or token_mask.device.type!='cpu' or token_mask.dtype!=torch.bool
            or token_mask.shape!=stream.shape[:2]):
        raise ValueError('Optional boolean CPU token mask must match [batch,tokens]')
    if any(not torch.isfinite(t).all() for t in (stream,rows,wkv_weight,query_weight,key_weight)):
        raise ValueError('Finite explicitly prepared CPU projection inputs required')
    projected=torch.nn.functional.linear(rows.flatten(-2),wkv_weight)
    key=projected[...,:copies*dim].reshape(batch,tokens,copies,dim)
    value=projected[...,copies*dim:]
    return reference_engram_residual(stream,key,value,query_weight,key_weight,
        norm_epsilon=norm_epsilon,signed_sqrt_floor=signed_sqrt_floor,token_mask=token_mask)
