"""Explicit selected-expert CUDA byte decode, not FP8 GEMM/native generation."""

from contextlib import ExitStack

from safetensors import safe_open
import torch

from .flash_next_fp8 import BlockFP8ExpertStore, FLOATING
from ..ops.flash_next_fp8 import NativeBlockFP8Decoder


class CUDABlockFP8ExpertStore(BlockFP8ExpertStore):
    def __init__(self, checkpoint, geometry, prefix, *, decoder, **budgets):
        if not isinstance(decoder, NativeBlockFP8Decoder):
            raise ValueError('Explicit prebuilt native FP8 byte decoder required')
        super().__init__(checkpoint, geometry, prefix, **budgets)
        self.decoder = decoder

    def read(self, expert, *, device, dtype):
        device = torch.device(device)
        if type(expert) is not int or not 0 <= expert < self.geometry.num_experts or dtype not in FLOATING or device.type != 'cuda':
            raise ValueError('Explicit CUDA device, floating dtype and valid selected expert required')
        parts = self._parts[expert]
        pairs = []
        with ExitStack() as stack:
            files = {str(s.file) for pair in parts for s in pair}
            readers = {path: stack.enter_context(safe_open(path, framework='pt', device='cpu')) for path in sorted(files)}
            for pair in parts:
                raw = []
                for spec in pair:
                    reader = readers[str(spec.file)]
                    view = reader.get_slice(spec.name)
                    if tuple(view.get_shape()) != spec.shape or view.get_dtype() != spec.dtype:
                        raise ValueError('CUDA FP8 expert backing metadata changed')
                    value = reader.get_tensor(spec.name)
                    raw.append(value.view(torch.uint8 if spec.dtype == 'F8_E4M3' else torch.int16).to(device=device))
                pairs.append(tuple(raw))
        outputs = self.decoder.decode_many(pairs, dtype=dtype)
        return torch.cat(outputs[:2], 0), outputs[2]
