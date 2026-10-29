"""CPU decoder for original-layout E4M3FN expert weights, not FP8 GEMM.

Official Flash-Next expert blocks use128x128 weight_scale_inv (BF16). It is a
DEQUANTIZATION multiplier despite its name. Decode FP32 products then cast to
explicit floating execution dtype. PLE global-scale rows use a separate codec.
No activation quantization, FP8 Tensor Core, INT8, cache/placement or native path.
"""

from contextlib import ExitStack

from safetensors import safe_open
import torch

from .flash_next_moe import MoEGeometry


FLOATING = (torch.float32, torch.float16, torch.bfloat16)


def decode_block_fp8(codes, scales, *, dtype, max_output_bytes=64*2**20,
                     strategy='tiled', max_work_bytes=64*2**20):
    """Own CPU output; tiled scratch or explicit bounded full-vector work budget."""
    if strategy not in ('tiled', 'vectorized') or type(max_work_bytes) is not int or max_work_bytes <= 0:
        raise ValueError('Explicit tiled/vectorized decoder and positive work budget required')
    if type(max_output_bytes) is not int or max_output_bytes <= 0 or dtype not in FLOATING:
        raise ValueError('Explicit positive decoded byte budget and floating dtype required')
    if codes.ndim != 2 or min(codes.shape) < 1 or codes.dtype != torch.float8_e4m3fn or (
            codes.device.type != 'cpu' or scales.device.type != 'cpu' or scales.dtype != torch.bfloat16):
        raise ValueError('CPU2D E4M3FN codes and BF16 block scales required')
    m, n = codes.shape
    if scales.shape != ((m+127)//128, (n+127)//128):
        raise ValueError('BF16 block-scale grid must match128x128 code geometry')
    size = torch.empty((), dtype=dtype, device='cpu').element_size()
    if m*n*size > max_output_bytes:
        raise ValueError('Decoded matrix exceeds explicit output byte budget')
    if not torch.isfinite(scales.float()).all() or not (scales.float() > 0).all():
        raise ValueError('Block dequantization multipliers must be positive finite')
    if strategy == 'vectorized' and m % 128 == 0 and n % 128 == 0:
        # Broadcast a tiny block grid over the logical128x128 view. Do NOT
        # repeat_interleave scales into a full MxN matrix. Full FP32 scratch is
        # deliberately admitted; non-multiple tails retain the tiled path.
        if m*n*(14+size) > max_work_bytes:
            raise ValueError('Vectorized FP8 decode exceeds explicit work byte budget')
        values = codes.float().contiguous()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite E4M3FN codes are not accepted')
        decoded = (values.view(m//128, 128, n//128, 128)*scales.float().view(m//128, 1, n//128, 1)).reshape(m, n)
        if not torch.isfinite(decoded).all():
            raise ValueError('Nonfinite decoded FP32 weight')
        decoded = decoded.to(dtype)
        if not torch.isfinite(decoded).all():
            raise ValueError('Decoded weight overflows selected floating execution dtype')
        return decoded
    output = torch.empty((m, n), dtype=dtype, device='cpu')
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            tile = codes[i*128:(i+1)*128, j*128:(j+1)*128].float()
            if not torch.isfinite(tile).all():
                raise ValueError('Nonfinite E4M3FN codes are not accepted')
            value = tile*scales[i, j].float()
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite decoded FP32 weight')
            target = output[i*128:(i+1)*128, j*128:(j+1)*128]
            target.copy_(value)
            if not torch.isfinite(target).all():
                raise ValueError('Decoded weight overflows selected floating execution dtype')
    return output


class BlockFP8ExpertStore:
    """Lazy selected expert gate/up/down -> floating matrices for existing MoE.

    Metadata of ALL configured experts is required; missing experts are errors.
    Only selected expert payloads are decoded. No GPU FP8 support required.
    Byte budgets cover serialized and decoded output, not total process peak.
    """

    def __init__(self, checkpoint, geometry, prefix, *, max_serialized_expert_bytes=16*2**20,
                 max_decoded_expert_bytes=64*2**20, decode_strategy='tiled', max_decode_work_bytes=64*2**20):
        if not isinstance(geometry, MoEGeometry):
            raise ValueError('Explicit qualified MoE geometry required')
        if decode_strategy not in ('tiled', 'vectorized') or type(max_decode_work_bytes) is not int or max_decode_work_bytes <= 0:
            raise ValueError('Explicit decoder strategy and positive work budget required')
        self.decode_strategy, self.max_decode_work_bytes = decode_strategy, max_decode_work_bytes
        config = checkpoint.config.get('quantization_config')
        if not isinstance(config, dict) or config.get('quant_method') != 'fp8' or (
                config.get('weight_block_size') != [128, 128] or config.get('weight_per_tensor') is not False or
                config.get('activation_scheme') != 'dynamic'):
            raise ValueError('Explicit original block128 FP8 checkpoint encoding required')
        if any(type(v) is not int or v <= 0 for v in (max_serialized_expert_bytes, max_decoded_expert_bytes)) or (
                geometry.num_experts > 4096):
            raise ValueError('Bounded explicit expert count and serialized/decoded byte budgets required')
        self.geometry, self._parts = geometry, []
        self.max_decoded_expert_bytes = max_decoded_expert_bytes
        g = geometry
        if 3*g.hidden_size*g.intermediate_size*4 > max_decoded_expert_bytes:
            raise ValueError('FP32 decoded selected expert exceeds explicit output byte budget')
        for e in range(g.num_experts):
            parts = []
            for projection, shape in (('gate_proj', (g.intermediate_size, g.hidden_size)),
                                      ('up_proj', (g.intermediate_size, g.hidden_size)),
                                      ('down_proj', (g.hidden_size, g.intermediate_size))):
                base = f'{prefix}.{e}.{projection}'
                weight, scale = checkpoint.tensors.get(base+'.weight'), checkpoint.tensors.get(base+'.weight_scale_inv')
                if weight is None or weight.shape != shape or weight.dtype != 'F8_E4M3' or scale is None or (
                        scale.shape != tuple((v+127)//128 for v in shape) or scale.dtype != 'BF16'):
                    raise ValueError('Original per-expert E4M3FN/BF16 block tensors required: '+base)
                parts.append((weight, scale))
            if sum(s.nbytes for pair in parts for s in pair) > max_serialized_expert_bytes:
                raise ValueError('Serialized selected expert exceeds explicit byte budget')
            self._parts.append(tuple(parts))
        self.serialized_bytes = sum(s.nbytes for parts in self._parts for pair in parts for s in pair)

    def read(self, expert, *, device, dtype):
        if type(expert) is not int or not 0 <= expert < self.geometry.num_experts or dtype not in FLOATING:
            raise ValueError('Valid selected expert and explicit floating dtype required')
        parts = self._parts[expert]
        output = []
        with ExitStack() as stack:
            files = {str(s.file) for pair in parts for s in pair}
            readers = {path: stack.enter_context(safe_open(path, framework='pt', device='cpu')) for path in sorted(files)}
            for pair in parts:
                tensors = []
                for spec in pair:
                    reader = readers[str(spec.file)]
                    view = reader.get_slice(spec.name)
                    if tuple(view.get_shape()) != spec.shape or view.get_dtype() != spec.dtype:
                        raise ValueError('FP8 expert backing metadata changed')
                    tensors.append(reader.get_tensor(spec.name))
                output.append(decode_block_fp8(*tensors, dtype=dtype, max_output_bytes=self.max_decoded_expert_bytes,
                    strategy=self.decode_strategy, max_work_bytes=self.max_decode_work_bytes))
        # Existing routed_experts expects gate then up concatenated; official
        # checkpoint has separate named projections. Never interleave the rows.
        return torch.cat(output[:2], 0).to(device=device), output[2].to(device=device)
